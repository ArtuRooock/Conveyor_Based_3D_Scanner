"""
Запуск расчёта объёма на .pts файле.

Формат файла: в каждой строке 4 числа:
    <индекс_среза>  <x>  <y>  <z>

Использование:
    python run_pts.py файл.pts            # диагностика + расчёт
    python run_pts.py файл.pts --info     # только диагностика
    python run_pts.py файл.pts --dump     # + сохранить точки после фильтров

Назначение фильтров и подбор параметров — в ALGORITHM.md, раздел 5.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from volume_core import ConveyorVolume

# ----------------------------- CONFIG -----------------------------

DIRECTION   = [1.0, 0.0, 0.0]   # ось, вдоль которой развёрнуто облако
BELT_NORMAL = [0.0, 0.0, 1.0]   # ось высоты
BELT_LEVEL  = -1.0              # уровень ленты; None = оценить автоматом
BELT_EPS    = 0.4               # точки ниже (лента + eps) выбрасываются
SLICE_STEP  = 0.3               # толщина слоя ds
CELL        = 0.1               # пиксель растра площади
MIN_GAP     = 1.0               # минимальный зазор между предметами
CLOSED_BOTTOM = True            # плоское дно на ленте
PROFILE_MODE  = True            # одна камера сверху: площадь по огибающей
MIN_VOLUME  = 1.0               # фигуры мельче этого - мусор, не показывать
TARGET_VOLUME = None            # целевой объём порции для поиска точек реза;
                                # None = резка отключена
MIN_BIN_FRAC = 0.10             # доля от медианного заполнения слоя, ниже
                                # которой слой считается пустым; 0 = откл.
DISPLACEMENT_PER_FRAME = 0.0    # 0, если облако уже развёрнуто вдоль движения;
                                # иначе шаг ленты на один индекс среза
W_MIN       = -55              # поперечные границы рабочей зоны ленты:
W_MAX       = 15              # точки вне (W_MIN..W_MAX) по оси поперёк
                                # ленты выбрасываются (отражения, фон);
                                # None = не ограничивать
VOXEL       = 0.5               # размер вокселя фильтра шума
MIN_VOXEL_PTS = 40              # порог точек в вокселе (ниже - шум);
                                # None = автомат, 0 = отключить

# -------------------------------------------------------------------


def load_pts(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("ожидался формат: индекс x y z (4 колонки)")
    frames = data[:, 0].astype(int)
    pts = data[:, 1:4]
    return frames, pts


def diagnose(frames, pts, u, n):
    print("=== ДИАГНОСТИКА ===")
    print("точек всего      : %d" % len(pts))
    print("срезов (кадров)  : %d  (индексы %d..%d)"
          % (len(np.unique(frames)), frames.min(), frames.max()))
    for name, col in zip("xyz", pts.T):
        print("  %s: min %10.3f  max %10.3f  размах %8.3f"
              % (name, col.min(), col.max(), col.max() - col.min()))

    # движется ли проекция на DIRECTION вместе с индексом среза?
    s = pts @ u
    lo, hi = frames.min(), frames.max()
    m0 = frames <= lo + (hi - lo) // 20
    m1 = frames >= hi - (hi - lo) // 20
    s0, s1 = s[m0].mean(), s[m1].mean()
    drift = (s1 - s0) / max(hi - lo, 1)
    print("проекция на DIRECTION: начало потока ~%.2f, конец ~%.2f" % (s0, s1))
    print("  -> дрейф %.5f на один индекс среза" % drift)
    if abs(s1 - s0) > 1.0:
        print("  -> координаты УЖЕ развёрнуты вдоль движения:")
        print("     DISPLACEMENT_PER_FRAME = 0")
    else:
        print("  -> координаты НЕ развёрнуты: каждый кадр - профиль на месте.")
        print("     Нужно задать DISPLACEMENT_PER_FRAME = шаг ленты на кадр!")

    # оценка уровня ленты: пик гистограммы в нижней половине высот
    h = pts @ n
    counts, edges = np.histogram(h, bins=300)
    half = len(counts) // 2
    peak = np.argmax(counts[:half])
    belt = 0.5 * (edges[peak] + edges[peak + 1])
    share = counts[peak] / len(pts) * 100.0
    print("высоты (проекция на BELT_NORMAL): min %.3f  max %.3f"
          % (h.min(), h.max()))
    print("  пик гистограммы внизу: h ~= %.3f (%.1f%% точек в пике)"
          % (belt, share))
    if share > 3.0:
        print("  -> в данных много точек самой ленты;")
        print("     BELT_LEVEL ~= %.3f, фильтр BELT_EPS обязателен" % belt)
    else:
        print("  -> точек ленты мало или нет; BELT_LEVEL ~= %.3f (проверить"
              " визуализацией plot_pts.py!)" % belt)
    print("===================")
    return belt


def denoise(pts, voxel, min_pts):
    """Выбрасывает точки из редких вокселей: шум висит одиночками,
    а поверхность после сканера даёт плотные воксели."""
    if min_pts is not None and min_pts <= 0:
        return np.ones(len(pts), dtype=bool)
    ijk = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(ijk, axis=0, return_inverse=True,
                            return_counts=True)
    if min_pts is None:
        min_pts = max(2, int(np.median(cnt) * 0.25))
        print("фильтр шума: автопорог = %d точек на воксель" % min_pts)
    return cnt[inv] >= min_pts


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    info_only = "--info" in sys.argv

    frames, pts = load_pts(path)
    u = np.asarray(DIRECTION, float); u /= np.linalg.norm(u)
    n = np.asarray(BELT_NORMAL, float); n /= np.linalg.norm(n)

    belt_auto = diagnose(frames, pts, u, n)
    if info_only:
        return

    belt = belt_auto if BELT_LEVEL is None else BELT_LEVEL

    # поперечный фильтр: отражения и фон живут за пределами ленты
    if W_MIN is not None or W_MAX is not None:
        v = np.cross(n, u); v /= np.linalg.norm(v)
        w = pts @ v
        keep = np.ones(len(pts), dtype=bool)
        if W_MIN is not None:
            keep &= w >= W_MIN
        if W_MAX is not None:
            keep &= w <= W_MAX
        print("поперечный фильтр: осталось %d из %d точек"
              % (keep.sum(), len(pts)))
        frames, pts = frames[keep], pts[keep]

    # фильтр шума: одиночные выбросы заполняют зазоры и склеивают предметы
    keep = denoise(pts, VOXEL, MIN_VOXEL_PTS)
    print("фильтр шума: осталось %d из %d точек" % (keep.sum(), len(pts)))
    frames, pts = frames[keep], pts[keep]

    # фильтр ленты: её точки тоже заполняют зазоры между предметами
    h = pts @ n - belt
    keep = h > BELT_EPS
    print("фильтр ленты: осталось %d из %d точек" % (keep.sum(), len(pts)))
    frames, pts = frames[keep], pts[keep]

    # фильтр редких слоёв: плотный мусор в зазорах на порядки реже,
    # чем поверхность деталей
    if MIN_BIN_FRAC > 0:
        sb = np.floor((pts @ u) / SLICE_STEP).astype(int)
        _, inv, cnt = np.unique(sb, return_inverse=True, return_counts=True)
        thr = max(3, int(np.median(cnt) * MIN_BIN_FRAC))
        keep = cnt[inv] >= thr
        print("фильтр редких слоёв (порог %d точек): осталось %d из %d"
              % (thr, keep.sum(), len(pts)))
        frames, pts = frames[keep], pts[keep]

    if "--dump" in sys.argv:
        stem = path.rsplit(".", 1)[0]
        np.savetxt(stem + "_filtered.pts", np.column_stack([frames, pts]),
                   fmt="%d %.6f %.6f %.6f")
        print("отфильтрованные точки -> %s_filtered.pts" % stem)

    def live_report(fig, vol):
        if vol >= MIN_VOLUME:
            print("Figure %d: V = %.1f mm^3" % (fig, vol))

    def cut_report(fig, vol, len_start, len_frame):
        print("Рез: фигура %d, V = %.1f mm^3, от начала фигуры %.2f мм, "
              "от последнего кадра %.2f мм" % (fig, vol, len_start, len_frame))

    integ = ConveyorVolume(u, belt_normal=n, belt_level=belt,
                           slice_step=SLICE_STEP, cell=CELL,
                           min_gap=MIN_GAP, closed_bottom=CLOSED_BOTTOM,
                           profile_mode=PROFILE_MODE, on_object=live_report,
                           target_volume=TARGET_VOLUME, on_cut=cut_report)

    # кадры подаются в порядке индексов срезов
    order = np.argsort(frames, kind="stable")
    frames, pts = frames[order], pts[order]
    starts = np.searchsorted(frames, np.unique(frames))
    bounds = np.append(starts, len(frames))
    for k in range(len(bounds) - 1):
        chunk = pts[bounds[k]:bounds[k + 1]]
        disp = float(frames[bounds[k]]) * DISPLACEMENT_PER_FRAME
        integ.add_frame(chunk, disp)

    vols = integ.finalize()
    good = {f: v for f, v in vols.items() if v >= MIN_VOLUME}
    print()
    print("Итого предметов: %d (отброшено как мусор: %d)"
          % (len(good), len(vols) - len(good)))
    print("%-8s %-22s %s" % ("Фигура", "положение по s", "объём, mm^3"))
    for f in sorted(good, key=lambda f: -integ.ranges.get(f, (0, 0))[0]):
        s0, s1 = integ.ranges.get(f, (float("nan"),) * 2)
        print("%-8d %8.1f .. %-8.1f   %10.1f" % (f, s1, s0, good[f]))


if __name__ == "__main__":
    main()