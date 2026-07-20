"""
Виртуальный эксперимент точности резки.

Берёт STL-модель, считает эталон (точный объём и координаты резов равного
объёма), затем превращает модель в облако точек "как со сканера" (шум,
кадры, движение по ленте), прогоняет через наш алгоритм с
target_volume = V/N и сравнивает резы программы с эталонными.

Использование:
    python virtual_experiment.py модель.stl N_КУСКОВ

Требует: numpy, scipy, numpy-stl + наши volume_core.py, stream_api.py,
reference_cuts.py в той же папке.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stl import mesh
from tools.reference_cuts import equal_volume_cuts
from stream_api import VolumeStream, CutParams, DetailParams


# --------------------------- CONFIG ---------------------------

N_POINTS   = 500_000    # сколько точек сэмплировать с поверхности
NOISE      = 0.0        # шум сканера, мм (СКО)
BELT_SPEED = 50.0       # мм/с
FPS        = 40.0       # кадров в секунду
CAM_ZONE   = 15.0       # ширина зоны обзора камер, мм
SLICE_STEP = 0.25        # толщина слоя алгоритма
CELL       = 0.1       # пиксель растра площади
# режим геометрии: для произвольной замкнутой модели контур строится
# по реальным точкам; для модели с плоским дном поставь True/True
CLOSED_BOTTOM = False
PROFILE_MODE  = False

# ---------------------------------------------------------------


def sample_surface(tris, n_points, rng):
    """Равномерные точки на поверхности меша (по площади треугольников)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(tris), size=n_points, p=areas / areas.sum())
    # равномерная барицентрическая выборка внутри треугольника
    r1 = np.sqrt(rng.uniform(0, 1, n_points))
    r2 = rng.uniform(0, 1, n_points)
    return (a[idx] * (1 - r1)[:, None]
            + b[idx] * (r1 * (1 - r2))[:, None]
            + c[idx] * (r1 * r2)[:, None])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    path, n_pieces = sys.argv[1], int(sys.argv[2])
    rng = np.random.default_rng(0)

    # --- эталон ---
    m = mesh.Mesh.from_file(path)
    tris = m.vectors.astype(float)
    v_true, cuts_x = equal_volume_cuts(tris, n_pieces)
    x_min = tris[:, :, 0].min()
    x_max = tris[:, :, 0].max()
    target = v_true / n_pieces
    # резы, отмеренные от переднего края (первым к камерам едет край x_max)
    ref_from_front = sorted(x_max - np.asarray(cuts_x))

    print("Эталон: V = %.2f мм^3, кусок = %.2f мм^3, длина модели %.2f мм"
          % (v_true, target, x_max - x_min))

    # --- облако точек и виртуальный конвейер ---
    cloud = sample_surface(tris, N_POINTS, rng)
    cloud[:, 2] -= cloud[:, 2].min()          # ставим модель на ленту (h=0)
    cloud[:, 0] -= x_max                      # передний край в x=0

    vs = VolumeStream([1.0, 0.0, 0.0], target_volume=target,
                      belt_level=0.0, slice_step=SLICE_STEP, cell=CELL,
                      closed_bottom=CLOSED_BOTTOM, profile_mode=PROFILE_MODE,
                      min_gap=30.0)

    got_cuts, got_details = [], []

    def handle(ev):
        for e in (ev if isinstance(ev, list) else [ev]):
            if isinstance(e, CutParams):
                got_cuts.append(e.len_from_start)
            elif isinstance(e, DetailParams):
                got_details.append(e.V)

    u = np.array([1.0, 0.0, 0.0])
    length = x_max - x_min
    t, dt = 0.0, 1.0 / FPS
    while BELT_SPEED * t <= length + CAM_ZONE + 60.0:
        disp = BELT_SPEED * t
        world = cloud + u * disp
        msk = (world[:, 0] >= 0.0) & (world[:, 0] < CAM_ZONE)
        if msk.any():
            frame = world[msk] + rng.normal(0.0, NOISE, (int(msk.sum()), 3))
        else:
            frame = np.empty((0, 3))
        ev = vs.add_pts(frame, disp)
        if ev is not None:
            handle(ev)
        t += dt
    for e in vs.finish():
        handle(e)

    # --- сравнение ---
    v_got = got_details[0] if got_details else float("nan")
    print("Программа: V = %.2f мм^3 (ошибка объёма %+.2f %%)"
          % (v_got, 100.0 * (v_got - v_true) / v_true))
    print()
    print("%-6s %-14s %-14s %s" % ("Рез", "эталон, мм", "программа, мм",
                                   "ошибка, мм"))
    for i in range(max(len(ref_from_front), len(got_cuts))):
        ref = ref_from_front[i] if i < len(ref_from_front) else float("nan")
        got = got_cuts[i] if i < len(got_cuts) else float("nan")
        print("%-6d %-14.3f %-14.3f %+8.3f" % (i + 1, ref, got, got - ref))
    if len(got_cuts) != len(ref_from_front):
        print("Внимание: число резов не совпало (эталон %d, программа %d)"
              % (len(ref_from_front), len(got_cuts)))


if __name__ == "__main__":
    main()