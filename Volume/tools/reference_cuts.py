"""
Эталон для виртуального эксперимента: читает STL-модель, считает её точный
объём и координаты резов, делящих модель на куски равного объёма.

Использование:
    python reference_cuts.py модель.stl N_КУСКОВ [ОСЬ]

ОСЬ - вдоль какой оси модель едет по ленте: x, y или z (по умолчанию x).
Смотрите габариты модели: длинная сторона обычно и есть ось движения.

Например: python reference_cuts.py fish.stl 4
выведет полный объём, целевой объём куска V/N и координаты N-1 резов
вдоль оси X (ось движения по ленте).

Как считается. Объём замкнутого меша - сумма знаковых объёмов тетраэдров
(начало координат + каждый треугольник): V = сумма( (a x b) . c ) / 6.
Функция объёма части модели левее плоскости x = t строится отсечением
треугольников этой плоскостью; координата реза равного объёма ищется
бисекцией по t.

Требует: numpy, numpy-stl.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stl import mesh


def signed_volume(tris):
    """Объём замкнутого меша по треугольникам (M, 3, 3)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum()) / 6.0


def clip_volume(tris, t, axis=0):
    """Объём части меша с x <= t.

    По теореме о дивергенции для поля F = (x - t, 0, 0):
        V(t) = интеграл (x - t) * nx dA
    по части поверхности с x <= t (крышку сечения строить не нужно:
    на плоскости x = t подынтегральное выражение равно нулю).
    Каждый треугольник отсекается плоскостью (Сазерленд-Ходжман),
    интеграл по многоугольнику равен (центроид_x - t) * площадь * nx.
    """
    total = 0.0
    for tri in tris:
        # единичная нормаль и её x-компонента
        nvec = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(nvec)
        if norm < 1e-15:
            continue
        nx = nvec[axis] / norm

        # отсечение треугольника полуплоскостью x <= t
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

        # площадь и интеграл (x - t) dA по многоугольнику (веером)
        for i in range(1, len(poly) - 1):
            a, b, c = poly[0], poly[i], poly[i + 1]
            area2 = np.linalg.norm(np.cross(b - a, c - a))
            xc = (a[axis] + b[axis] + c[axis]) / 3.0
            total += nx * (xc - t) * area2 / 2.0
    return total


def equal_volume_cuts(tris, n_pieces, tol=1e-6, axis=0):
    """Координаты x резов, делящих модель на n_pieces равных по объёму."""
    v_total = abs(signed_volume(tris))
    x_min = tris[:, :, axis].min()
    x_max = tris[:, :, axis].max()
    sign = 1.0 if signed_volume(tris) > 0 else -1.0

    cuts = []
    for k in range(1, n_pieces):
        target = v_total * k / n_pieces
        lo, hi = x_min, x_max
        while hi - lo > tol * (x_max - x_min):
            mid = 0.5 * (lo + hi)
            if sign * clip_volume(tris, mid, axis) < target:
                lo = mid
            else:
                hi = mid
        cuts.append(0.5 * (lo + hi))
    return v_total, cuts


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    path, n = sys.argv[1], int(sys.argv[2])
    axis_name = sys.argv[3].lower() if len(sys.argv) > 3 else "x"
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]

    m = mesh.Mesh.from_file(path)
    tris = m.vectors.astype(float)          # (M, 3, 3)

    v_total, cuts = equal_volume_cuts(tris, n, axis=axis)
    x_min = tris[:, :, axis].min()
    x_max = tris[:, :, axis].max()

    print("Модель: %s" % path)
    print("Треугольников: %d" % len(tris))
    print("Габариты: X %.2f, Y %.2f, Z %.2f мм"
          % tuple(tris[:, :, i].max() - tris[:, :, i].min() for i in range(3)))
    print("Ось движения: %s, габарит %.3f .. %.3f (длина %.3f мм)"
          % (axis_name.upper(), x_min, x_max, x_max - x_min))
    print("Полный объём:  %.3f мм^3" % v_total)
    print("Кусков: %d, целевой объём куска: %.3f мм^3" % (n, v_total / n))
    print()
    print("Эталонные резы (координата по оси %s и расстояние от начала):"
          % axis_name.upper())
    for i, c in enumerate(cuts, 1):
        print("  рез %d: %s = %10.4f   от начала = %8.4f мм"
              % (i, axis_name, c, c - x_min))


if __name__ == "__main__":
    main()