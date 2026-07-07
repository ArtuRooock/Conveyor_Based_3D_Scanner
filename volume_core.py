"""
Итеративный расчёт объёма предметов на конвейере по потоку кадров облака точек.
Подробное описание алгоритма, формул и параметров — в ALGORITHM.md.

Требует: numpy, scipy.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


class LaserScanner:
    def __init__(self, f, cx, cy, b, eta, theta, R=None, t=None):
        self.f = float(f)
        self.cx = float(cx)
        self.cy = float(cy)
        self.b = float(b)

        # нормаль плоскости лазера: N = Ry(theta) * Rz(eta) * [1, 0, 0]
        ce, se = np.cos(eta), np.sin(eta)
        ct, st = np.cos(theta), np.sin(theta)
        Rz = np.array([[ce, -se, 0.0],
                       [se,  ce, 0.0],
                       [0.0, 0.0, 1.0]])
        Ry = np.array([[ct, 0.0, st],
                       [0.0, 1.0, 0.0],
                       [-st, 0.0, ct]])
        self.N = Ry @ Rz @ np.array([1.0, 0.0, 0.0])

        self.R = np.eye(3) if R is None else np.asarray(R, dtype=float)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=float)

    def pixels_to_points(self, uv):
        """uv -- (N, 2) пиксели центра лазерной линии.
        Возвращает (N, 3) точки в мировой системе координат."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        # луч из оптического центра через пиксель
        rays = np.column_stack([uv[:, 0] - self.cx,
                                uv[:, 1] - self.cy,
                                np.full(len(uv), self.f)])
        # пересечение луча с плоскостью лазера
        denom = rays @ self.N
        ok = np.abs(denom) > 1e-12          # луч параллелен плоскости -> брак
        scale = (self.b * self.N[0]) / denom[ok]
        pts_cam = rays[ok] * scale[:, None]
        # перевод из системы камеры в мировую
        return pts_cam @ self.R.T + self.t


class _TrackedObject:
    # один предмет на ленте: его слои и накопленный объём

    __slots__ = ("fig_id", "bins", "min_idx", "max_idx", "volume")

    def __init__(self, fig_id):
        self.fig_id = fig_id
        self.bins = {}           # индекс слоя -> список [(w, h), ...]
        self.min_idx = None
        self.max_idx = None
        self.volume = 0.0


class ConveyorVolume:
    # итеративный счётчик объёма предметов на конвейере

    def __init__(self, direction, belt_normal=(0.0, 0.0, 1.0),
                 belt_level=0.0, slice_step=1.0, close_margin=2,
                 closed_bottom=True, cell=0.25,
                 min_gap=30.0, unit="mm", on_object=None,
                 profile_mode=False):
        
        u = np.asarray(direction, dtype=float)
        u = u / np.linalg.norm(u)
        n = np.asarray(belt_normal, dtype=float)
        n = n / np.linalg.norm(n)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        self.u, self.v, self.n = u, v, n
        self.belt_level = float(belt_level)
        self.ds = float(slice_step)
        self.margin = int(close_margin)
        self.closed_bottom = bool(closed_bottom)
        self.cell = float(cell)
        self.profile_mode = bool(profile_mode)
        self.gap_bins = max(int(np.ceil(float(min_gap) / self.ds)), 1)
        self.unit = str(unit)
        self.on_object = on_object or self._default_report

        self._objects = []       # активные предметы (в кадре может быть два)
        self._fig_counter = 0
        self._cam_top = -np.inf  # верхняя граница зоны камер (мировая коорд.)
        self.volumes = {}        # fig_id -> готовый объём
        self.ranges = {}         # fig_id -> (s_min, s_max) по ленте

    # ------------------- публичный API -------------------

    def add_frame(self, points, displacement):
        # обработать очередной кадр
        # points - (N, 3) точки в мировой системе; пустой массив тоже допустим (продвинет окно и закроет уехавшие слои)
        # displacement - путь, пройденный лентой к моменту кадра

        displacement = float(displacement)
        p = np.asarray(points, dtype=float).reshape(-1, 3)

        if p.size:
            s_world = p @ self.u
            self._cam_top = max(self._cam_top, float(s_world.max()))

            s = s_world - displacement            # ленточная координата
            w = p @ self.v                        # поперёк ленты
            h = np.clip(p @ self.n - self.belt_level, 0.0, None)
            idx = np.floor(s / self.ds).astype(int)

            for cluster in self._split_clusters(idx):
                mask = np.isin(idx, cluster)
                obj = self._match_object(cluster)
                for i, wi, hi in zip(idx[mask], w[mask], h[mask]):
                    obj.bins.setdefault(int(i), []).append((wi, hi))
                lo, hi_ = int(cluster.min()), int(cluster.max())
                obj.min_idx = lo if obj.min_idx is None else min(obj.min_idx, lo)
                obj.max_idx = hi_ if obj.max_idx is None else max(obj.max_idx, hi_)

        # всё, что выше окна камер, больше не пополнится - закрываем
        if np.isfinite(self._cam_top):
            top_idx = int(np.floor((self._cam_top - displacement) / self.ds))
            self._advance(top_idx + self.margin)

    def finalize(self):
        """Принудительно завершить все активные предметы (конец потока).
        Возвращает словарь fig_id -> объём."""
        for obj in list(self._objects):
            self._finish_object(obj)
        return dict(self.volumes)

    # ------------------- внутреннее -------------------

    def _default_report(self, fig_id, volume):
        print("Figure %d: V = %.1f %s^3" % (fig_id, volume, self.unit))

    def _split_clusters(self, idx):
        """Разбить индексы слоёв кадра на группы, разделённые зазором."""
        uniq = np.unique(idx)
        breaks = np.where(np.diff(uniq) > self.gap_bins)[0]
        return np.split(uniq, breaks + 1)

    def _match_object(self, cluster):
        """Найти предмет, к которому относится группа слоёв, или завести новый."""
        lo, hi = int(cluster.min()), int(cluster.max())
        for obj in self._objects:
            if (lo <= obj.max_idx + self.gap_bins and
                    hi >= obj.min_idx - self.gap_bins):
                return obj
        self._fig_counter += 1
        obj = _TrackedObject(self._fig_counter)
        self._objects.append(obj)
        return obj

    def _advance(self, closed_above):
        """Закрыть уехавшие слои; завершить полностью уехавшие предметы."""
        for obj in list(self._objects):
            for i in [k for k in obj.bins if k > closed_above]:
                pts = np.asarray(obj.bins.pop(i), dtype=float)
                obj.volume += self._section_area(pts) * self.ds
            if obj.min_idx is not None and obj.min_idx > closed_above:
                self._finish_object(obj)

    def _finish_object(self, obj):
        for i in list(obj.bins):
            pts = np.asarray(obj.bins.pop(i), dtype=float)
            obj.volume += self._section_area(pts) * self.ds
        self._objects.remove(obj)
        self.volumes[obj.fig_id] = obj.volume
        if obj.min_idx is not None:
            self.ranges[obj.fig_id] = (obj.min_idx * self.ds,
                                       (obj.max_idx + 1) * self.ds)
        self.on_object(obj.fig_id, obj.volume)

    def _section_area(self, pts):
        """Площадь сечения в плоскости (w, h)."""
        if len(pts) < 3:
            return 0.0
        cell = self.cell

        if self.profile_mode:
            return self._profile_area(pts, cell)

        if self.closed_bottom:
            # замыкаем контур по ленте
            base = pts.copy()
            base[:, 1] = 0.0
            pts = np.vstack([pts, base])
        else:
            # мостим невидимую щель у точки касания хордой
            h0 = pts[:, 1].min()
            band = pts[pts[:, 1] < h0 + 4 * cell]
            wl, wr = band[:, 0].min(), band[:, 0].max()
            if wr - wl > 2 * cell:
                t = np.linspace(0.0, 1.0, int((wr - wl) / cell) + 2)
                bridge = np.column_stack([wl + t * (wr - wl),
                                          np.full(len(t), h0)])
                pts = np.vstack([pts, bridge])

        # очистка от одиночных точек-выбросов
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = d[:, 1]
        lim = max(4.0 * np.percentile(nn, 95), 4.0 * cell)
        good = nn <= lim
        if good.sum() >= 3:
            pts, nn = pts[good], nn[good]

        # стартовый радиус замыкания и его потолок
        k = max(int(np.ceil(nn.max() / (2.0 * cell))) + 1, 1)
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        k_cap = max(k, int(np.ceil(0.15 * span / cell)))

        lo = pts.min(axis=0) - (k_cap + 2) * cell
        ij = np.floor((pts - lo) / cell).astype(int)
        img = np.zeros(ij.max(axis=0) + k_cap + 3, dtype=bool)
        img[ij[:, 0], ij[:, 1]] = True

        # адаптивное замыкание: растим радиус, пока контур не сомкнётся
        st = ndimage.generate_binary_structure(2, 2)
        img = ndimage.binary_dilation(img, st, iterations=k)
        while True:
            filled = ndimage.binary_fill_holes(img)
            if (filled.sum() - img.sum() > 0.01 * img.sum()) or k >= k_cap:
                break
            grow = min(k, k_cap - k)
            img = ndimage.binary_dilation(img, st, iterations=grow)
            k += grow
        img = ndimage.binary_erosion(filled, st, iterations=k)
        return float(img.sum()) * cell * cell

    @staticmethod
    def _profile_area(pts, cell):
        """Интеграл под верхней огибающей профиля (режим одной камеры)."""
        wb = np.floor(pts[:, 0] / cell).astype(int)
        lo = wb.min()
        hmax = np.full(wb.max() - lo + 1, -1.0)
        np.maximum.at(hmax, wb - lo, pts[:, 1])
        have = hmax >= 0.0
        xs = np.arange(len(hmax), dtype=float)
        hmax = np.interp(xs, xs[have], hmax[have])
        return float(hmax.sum()) * cell