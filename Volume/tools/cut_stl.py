"""
Разрезание STL-модели плоскостью с закрытием среза плоской крышкой.
Результат - водонепроницаемая половина модели с плоским дном,
готовая к 3D-печати и к нашему виртуальному/натурному эксперименту.

Использование:
    python cut_stl.py модель.stl                        # верх, равные объёмы, ось Z
    python cut_stl.py модель.stl --keep lower           # нижняя половина
    python cut_stl.py модель.stl --axis y --pos middle  # по середине габарита
    python cut_stl.py модель.stl --pos 12.5             # по координате 12.5

Параметры:
    --axis x|y|z     ось, перпендикулярная плоскости реза (по умолчанию z)
    --pos volume|middle|ЧИСЛО
                     положение плоскости: 'volume' = равные объёмы половин
                     (по умолчанию), 'middle' = середина габарита, либо
                     координата напрямую
    --keep upper|lower   какую половину оставить (по умолчанию upper)

Оставленная половина кладётся плоскостью реза вниз (сдвигается в ноль по
оси реза), чтобы дно печаталось прямо на столе принтера.

Требует: numpy, scipy, numpy-stl, shapely.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stl import mesh
import mapbox_earcut
from shapely.geometry import Polygon


def signed_volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum()) / 6.0


def clip_volume_axis(tris, axis, t):
    """Объём части меша с координатой по оси <= t (см. reference_cuts)."""
    total = 0.0
    for tri in tris:
        nvec = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(nvec)
        if norm < 1e-15:
            continue
        na = nvec[axis] / norm
        poly = []
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            pin, qin = p[axis] <= t, q[axis] <= t
            if pin:
                poly.append(p)
            if pin != qin:
                s = (t - p[axis]) / (q[axis] - p[axis])
                poly.append(p + s * (q - p))
        if len(poly) < 3:
            continue
        poly = np.asarray(poly, dtype=float)
        for i in range(1, len(poly) - 1):
            a, b, c = poly[0], poly[i], poly[i + 1]
            area2 = np.linalg.norm(np.cross(b - a, c - a))
            xc = (a[axis] + b[axis] + c[axis]) / 3.0
            total += na * (xc - t) * area2 / 2.0
    return total


def find_plane(tris, axis, pos):
    lo = tris[:, :, axis].min()
    hi = tris[:, :, axis].max()
    if pos == "middle":
        return 0.5 * (lo + hi)
    if pos != "volume":
        return float(pos)
    # бисекция до равных объёмов
    v_total = abs(signed_volume(tris))
    sign = 1.0 if signed_volume(tris) > 0 else -1.0
    a, b = lo, hi
    while b - a > 1e-6 * (hi - lo):
        m = 0.5 * (a + b)
        if sign * clip_volume_axis(tris, axis, m) < v_total / 2.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def cut_mesh(tris, axis, t, keep_upper):
    """Отсекает половину меша плоскостью и закрывает срез крышкой.
    Возвращает массив треугольников (M, 3, 3)."""
    if keep_upper:
        # переворачиваем задачу: оставить coord >= t
        flip = tris.copy()
        flip[:, :, axis] = -flip[:, :, axis]
        flip = flip[:, ::-1, :]              # сохранить ориентацию наружу
        out = cut_mesh(flip, axis, -t, keep_upper=False)
        out[:, :, axis] = -out[:, :, axis]
        return out[:, ::-1, :]

    kept = []       # треугольники оставленной части
    segs = []       # отрезки линии среза (для крышки)
    o1, o2 = [a for a in (0, 1, 2) if a != axis]

    for tri in tris:
        poly, cross = [], []
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            pin, qin = p[axis] <= t, q[axis] <= t
            if pin:
                poly.append(p)
            if pin != qin:
                s = (t - p[axis]) / (q[axis] - p[axis])
                ip = p + s * (q - p)
                poly.append(ip)
                cross.append(ip)
        if len(poly) >= 3:
            poly = np.asarray(poly)
            for i in range(1, len(poly) - 1):
                kept.append(np.stack([poly[0], poly[i], poly[i + 1]]))
        if len(cross) == 2:
            segs.append([(cross[0][o1], cross[0][o2]),
                         (cross[1][o1], cross[1][o2])])

    # крышка: собираем контур среза из отрезков и триангулируем.
    # Кольца собираются вручную обходом отрезков (без пересборки координат),
    # триангуляция ear-clipping уважает каждый узел контура - шов герметичен.
    if segs:
        key = lambda p: (round(p[0], 6), round(p[1], 6))
        adj = {}
        for a, b in segs:
            adj.setdefault(key(a), []).append((a, b))
            adj.setdefault(key(b), []).append((b, a))
        used = set()
        loops = []
        for a, b in segs:
            if (key(a), key(b)) in used:
                continue
            loop = [np.asarray(a)]
            cur, prev_k = b, key(a)
            used.add((key(a), key(b))); used.add((key(b), key(a)))
            guard = 0
            while key(cur) != key(loop[0]) and guard < 100000:
                loop.append(np.asarray(cur))
                nxts = [q for p, q in adj.get(key(cur), [])
                        if (key(cur), key(q)) not in used and key(q) != key(p)]
                if not nxts:
                    break
                nxt = nxts[0]
                used.add((key(cur), key(nxt))); used.add((key(nxt), key(cur)))
                cur = nxt
                guard += 1
            if key(cur) == key(loop[0]) and len(loop) >= 3:
                loops.append(np.asarray(loop))

        # сортируем по площади: самые большие - внешние контуры,
        # кольцо внутри другого кольца - дырка в крышке
        loops.sort(key=lambda L: -abs(Polygon(L).area))
        outers = []
        for L in loops:
            pg = Polygon(L)
            if not pg.is_valid or pg.area < 1e-9:
                continue
            placed = False
            for entry in outers:
                if entry["poly"].contains(pg.representative_point()):
                    entry["holes"].append(L)
                    placed = True
                    break
            if not placed:
                outers.append({"poly": pg, "loop": L, "holes": []})

        for entry in outers:
            rings = [entry["loop"]] + entry["holes"]
            verts = np.vstack(rings).astype(np.float64)
            ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
            tri_idx = mapbox_earcut.triangulate_float64(verts, ring_ends)
            for j in range(0, len(tri_idx), 3):
                a2, b2, c2 = verts[tri_idx[j]], verts[tri_idx[j+1]], verts[tri_idx[j+2]]
                tri3 = np.zeros((3, 3))
                for k, v2 in enumerate((a2, b2, c2)):
                    tri3[k, o1], tri3[k, o2] = v2
                    tri3[k, axis] = t
                nvec = np.cross(tri3[1] - tri3[0], tri3[2] - tri3[0])
                if nvec[axis] < 0:
                    tri3 = tri3[::-1]
                kept.append(tri3)

    return np.asarray(kept)


def check_watertight(tris):
    """Каждое ребро должно принадлежать ровно двум треугольникам."""
    edges = {}
    r = np.round(tris, 6)
    for tri in r:
        for i in range(3):
            e = tuple(sorted((tuple(tri[i]), tuple(tri[(i + 1) % 3]))))
            edges[e] = edges.get(e, 0) + 1
    bad = sum(1 for v in edges.values() if v != 2)
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    axis = {"x": 0, "y": 1, "z": 2}[_arg("--axis", "z").lower()]
    pos = _arg("--pos", "volume")
    keep_upper = _arg("--keep", "upper").lower() == "upper"

    m = mesh.Mesh.from_file(path)
    tris = m.vectors.astype(float)
    v_full = abs(signed_volume(tris))

    t = find_plane(tris, axis, pos)
    # если плоскость попала точно в вершины меша (вырожденный случай),
    # сдвигаем её на волосок
    span = tris[:, :, axis].max() - tris[:, :, axis].min()
    if np.min(np.abs(tris[:, :, axis] - t)) < 1e-9 * max(span, 1.0):
        t += 1e-5 * span
    print("Модель: %s, объём %.2f мм^3" % (path, v_full))
    print("Плоскость реза: ось %s, координата %.4f" % ("xyz"[axis], t))

    half = cut_mesh(tris, axis, t, keep_upper)
    half[:, :, axis] -= half[:, :, axis].min()   # плоскость реза в ноль

    v_half = abs(signed_volume(half))
    bad = check_watertight(half)
    print("Половина: %d треугольников, объём %.2f мм^3 (%.1f%% от целой)"
          % (len(half), v_half, 100.0 * v_half / v_full))
    print("Проверка водонепроницаемости: %s"
          % ("ОК" if bad == 0 else "ПЛОХИХ РЁБЕР: %d" % bad))

    out = path.rsplit(".", 1)[0] + "_half.stl"
    data = np.zeros(len(half), dtype=mesh.Mesh.dtype)
    data["vectors"] = half.astype(np.float32)
    mesh.Mesh(data).save(out)
    print("Сохранено: %s" % out)


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    main()