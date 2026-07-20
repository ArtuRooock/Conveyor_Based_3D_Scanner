"""
Синтетические тесты алгоритма из volume_core.py: фигуры с известным объёмом
прогоняются через виртуальный конвейер, результат сравнивается с аналитикой.

Запуск: python conveyor_volume.py
Запускать после любых правок volume_core.py - ошибки должны остаться
в пределах нескольких процентов.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from volume_core import ConveyorVolume



def run_conveyor(cloud_obj, integ, belt_speed=50.0, fps=40.0,
                 cam_zone=15.0, noise=0.1, seed=1):
    """Прогоняет готовое облако предмета через виртуальный конвейер:
    предмет едет мимо неподвижной зоны камер, кадры идут с частотой fps."""
    rng = np.random.default_rng(seed)
    length = cloud_obj[:, 0].max() - cloud_obj[:, 0].min()
    t, dt = 0.0, 1.0 / fps
    while belt_speed * t <= length + cam_zone + 60.0:
        disp = belt_speed * t
        world = cloud_obj + integ.u * disp
        mask = (world[:, 0] >= 0.0) & (world[:, 0] < cam_zone)
        if mask.any():
            frame = world[mask] + rng.normal(0.0, noise,
                                             (int(mask.sum()), 3))
            integ.add_frame(frame, disp)
        else:
            integ.add_frame(np.empty((0, 3)), disp)
        t += dt
    return integ.finalize()


def report(name, vols, v_true):
    v = sum(vols.values())
    print("%-14s %10.0f / %10.0f  (ошибка %5.2f %%)"
          % (name, v, v_true, 100.0 * abs(v - v_true) / v_true))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    direction = [1.0, 0.0, 0.0]
    quiet = lambda fig, vol: None      # без живой печати, только итог

    # --- Тест 1: полуцилиндр на плоской стороне, V = pi R^2 / 2 * L ---
    R, L = 30.0, 200.0
    N = 400_000
    x = rng.uniform(-L, 0.0, N)
    phi = rng.uniform(0.0, np.pi, N)
    cloud = np.column_stack([x, R * np.cos(phi), R * np.sin(phi)])
    vols = run_conveyor(cloud, ConveyorVolume(direction, closed_bottom=True,
                                              on_object=quiet))
    report("Полуцилиндр", vols, np.pi * R * R / 2.0 * L)

    # --- Тест 2: шар на ленте, V = 4/3 pi R^3 ---
    # камеры не видят нижние 2 мм у точки касания
    R = 40.0
    N = 800_000
    uv = rng.normal(size=(N, 3))
    uv /= np.linalg.norm(uv, axis=1, keepdims=True)
    sphere = uv * R + np.array([-R, 0.0, R])
    sphere = sphere[sphere[:, 2] > 2.0]
    vols = run_conveyor(sphere, ConveyorVolume(direction, closed_bottom=False,
                                               slice_step=0.5,
                                               on_object=quiet))
    report("Шар", vols, 4.0 / 3.0 * np.pi * R ** 3)

    # --- Тест 3: П-образный профиль с пазом (невыпуклое сечение) ---
    # внешний прямоугольник 40x30, паз сверху 16x20, длина 200
    W, H, gw, gd, L = 40.0, 30.0, 16.0, 20.0, 200.0
    contour = [(-W/2, 0), (-W/2, H), (-gw/2, H), (-gw/2, H - gd),
               (gw/2, H - gd), (gw/2, H), (W/2, H), (W/2, 0)]
    segs = []
    for (w0, h0), (w1, h1) in zip(contour[:-1], contour[1:]):
        tp = rng.uniform(0.0, 1.0, 40_000)
        segs.append(np.column_stack([w0 + tp * (w1 - w0),
                                     h0 + tp * (h1 - h0)]))
    wh = np.vstack(segs)
    xs = rng.uniform(-L, 0.0, len(wh))
    cloud = np.column_stack([xs, wh[:, 0], wh[:, 1]])
    vols = run_conveyor(cloud, ConveyorVolume(direction, closed_bottom=True,
                                              on_object=quiet))
    report("П-профиль", vols, (W * H - gw * gd) * L)

    # --- Тест 4: разделение потока из трёх предметов + profile_mode ---
    # коробка, полуцилиндр и П-профиль друг за другом с зазором 60 мм
    GAP = 60.0
    clouds = []

    Lb, Wb, Hb = 100.0, 60.0, 40.0            # коробка: виден верх и бока
    n = 200_000
    top = np.column_stack([rng.uniform(-Lb, 0, n),
                           rng.uniform(-Wb/2, Wb/2, n),
                           np.full(n, Hb)])
    side = np.column_stack([rng.uniform(-Lb, 0, n),
                            rng.choice([-Wb/2, Wb/2], n),
                            rng.uniform(0, Hb, n)])
    clouds.append(np.vstack([top, side]))

    Rc, Lc = 30.0, 150.0                      # полуцилиндр
    n = 300_000
    phi = rng.uniform(0, np.pi, n)
    clouds.append(np.column_stack([rng.uniform(-Lc, 0, n),
                                   Rc*np.cos(phi), Rc*np.sin(phi)]))

    Wp, Hp, gw, gd, Lp = 40.0, 30.0, 16.0, 20.0, 120.0   # П-профиль
    contour = [(-Wp/2, 0), (-Wp/2, Hp), (-gw/2, Hp), (-gw/2, Hp-gd),
               (gw/2, Hp-gd), (gw/2, Hp), (Wp/2, Hp), (Wp/2, 0)]
    segs = []
    for (w0, h0), (w1, h1) in zip(contour[:-1], contour[1:]):
        tp = rng.uniform(0, 1, 30_000)
        segs.append(np.column_stack([w0 + tp*(w1-w0), h0 + tp*(h1-h0)]))
    wh = np.vstack(segs)
    clouds.append(np.column_stack([rng.uniform(-Lp, 0, len(wh)),
                                   wh[:, 0], wh[:, 1]]))

    # расставляем предметы друг за другом по ленте
    scene = []
    front = 0.0
    for c in clouds:
        c = c.copy()
        c[:, 0] += front - c[:, 0].max()
        scene.append(c)
        front = c[:, 0].min() - GAP
    scene = np.vstack(scene)

    print()
    print("Поток из трёх предметов (живая печать по мере прохода камер):")
    integ = ConveyorVolume(direction, closed_bottom=True, min_gap=30.0,
                           profile_mode=True)
    run_conveyor(scene, integ)
    print("Ожидаемые значения:")
    print("  Figure 1 (коробка)     : %.0f" % (Lb * Wb * Hb))
    print("  Figure 2 (полуцилиндр) : %.0f" % (np.pi * Rc * Rc / 2 * Lc))
    print("  Figure 3 (П-профиль)   : %.0f" % ((Wp * Hp - gw * gd) * Lp))