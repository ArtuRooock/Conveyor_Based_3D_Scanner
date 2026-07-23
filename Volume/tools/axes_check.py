"""
axes_check.py - определяет по облаку точек, какая ось что означает.

Отвечает на вопросы: вдоль какой оси едет лента, какая ось - высота,
на каком уровне лежит поверхность ленты. Нужен, когда облако приходит
от нового сканера и соглашение об осях неизвестно.

Использование:
    python axes_check.py облако.pts
    python axes_check.py облако.pts 10       # с пересчётом единиц (см -> мм)

Файл: 3 колонки "x y z" или 4 колонки "индекс x y z".
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def load(path, scale):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] not in (3, 4):
        raise ValueError("ожидался формат 'x y z' или 'индекс x y z'")
    pts = (data[:, 1:4] if data.shape[1] == 4 else data) * scale
    frames = data[:, 0].astype(int) if data.shape[1] == 4 else None
    return frames, pts


def flatness(values, bins=400):
    """Насколько ось похожа на 'высоту': есть ли узкий плотный слой.

    У оси высоты лента даёт резкий пик (много точек в узком диапазоне).
    Возвращает долю точек в самом населённом узком слое и его положение.
    """
    hist, edges = np.histogram(values, bins=bins)
    top = int(np.argmax(hist))
    # объединяем соседние корзины, чтобы не зависеть от ширины
    window = hist[max(0, top - 1):top + 2].sum()
    center = 0.5 * (edges[top] + edges[top + 1])
    return window / len(values), center


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "cloud.pts"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    frames, pts = load(path, scale)
    names = ["x", "y", "z"]

    print("=== ОСИ ОБЛАКА ===")
    print("точек: %d, масштаб координат: x%.3g" % (len(pts), scale))
    print("")

    ranges = []
    print("%-3s %10s %10s %10s   %s" % ("ось", "min", "max", "размах",
                                        "доля точек в плотном слое"))
    for i in range(3):
        values = pts[:, i]
        span = values.max() - values.min()
        frac, center = flatness(values)
        ranges.append(span)
        print("%-3s %10.2f %10.2f %10.2f   %5.1f%%  (слой около %.2f)"
              % (names[i], values.min(), values.max(), span,
                 100 * frac, center))

    move_axis = int(np.argmax(ranges))
    print("")
    print("Ось движения: скорее всего %s - у неё наибольший размах (%.1f)"
          % (names[move_axis], ranges[move_axis]))

    # высота: из двух оставшихся осей та, где плотный слой выражен сильнее
    rest = [i for i in range(3) if i != move_axis]
    fr0, c0 = flatness(pts[:, rest[0]])
    fr1, c1 = flatness(pts[:, rest[1]])
    if fr0 >= fr1:
        height_axis, belt_level, other = rest[0], c0, rest[1]
    else:
        height_axis, belt_level, other = rest[1], c1, rest[0]

    print("Ось высоты:   скорее всего %s - лента даёт плотный слой на %.2f"
          % (names[height_axis], belt_level))
    print("Поперёк ленты: %s (размах %.1f - это ширина рабочей зоны)"
          % (names[other], ranges[other]))

    # проверка: сколько точек выше ленты (это и есть детали)
    h = pts[:, height_axis] - belt_level
    for eps in (0.5, 1.0, 2.0, 5.0):
        print("  выше ленты на %.1f: %6d точек (%.1f%%)"
              % (eps, (h > eps).sum(), 100 * (h > eps).mean()))

    print("")
    print("Строки для CONFIG в run_pts.py:")
    vec = lambda i: "[%s]" % ", ".join("1.0" if j == i else "0.0"
                                       for j in range(3))
    print("    DIRECTION   = %s" % vec(move_axis))
    print("    BELT_NORMAL = %s" % vec(height_axis))
    print("    BELT_LEVEL  = %.2f" % belt_level)

    # если высота уходит вниз - возможно, ось направлена в другую сторону
    above = (h > 1.0).sum()
    below = (h < -1.0).sum()
    if below > above:
        print("")
        print("ВНИМАНИЕ: точек НИЖЕ найденного уровня больше, чем выше")
        print("  (%d против %d). Вероятно, ось высоты смотрит вниз -" 
              % (below, above))
        print("  попробуйте BELT_NORMAL = %s"
              % ("[%s]" % ", ".join("-1.0" if j == height_axis else "0.0"
                                    for j in range(3))))


if __name__ == "__main__":
    main()
