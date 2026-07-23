"""
profile_check.py - показывает, что лежит вдоль ленты: где объекты,
где пусто, где мусор. Печатает текстовый профиль, картинки не нужны.

Использование:
    python profile_check.py облако.pts [масштаб] [ось_движения] [ось_высоты]

Например:
    python profile_check.py fish.pts 10 x y

Оси задаются буквами x/y/z (по умолчанию движение x, высота y).
Файл: 3 колонки "x y z" или 4 колонки "индекс x y z".
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def load(path, scale):
    data = np.loadtxt(path)
    pts = (data[:, 1:4] if data.shape[1] == 4 else data) * scale
    return pts


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "cloud.pts"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    move_ax = AXES[sys.argv[3]] if len(sys.argv) > 3 else 0
    high_ax = AXES[sys.argv[4]] if len(sys.argv) > 4 else 1
    cross_ax = [i for i in (0, 1, 2) if i not in (move_ax, high_ax)][0]

    pts = load(path, scale)
    s = pts[:, move_ax]
    h = pts[:, high_ax]
    w = pts[:, cross_ax]

    # уровень ленты: самый населённый узкий слой по высоте
    hist, edges = np.histogram(h, bins=400)
    top = int(np.argmax(hist))
    belt = 0.5 * (edges[top] + edges[top + 1])
    print("уровень ленты по высоте: %.2f (%.1f%% точек в слое)"
          % (belt, 100.0 * hist[top] / len(h)))

    above = h - belt > 1.0
    print("точек выше ленты на 1 мм: %d из %d (%.1f%%)"
          % (above.sum(), len(h), 100.0 * above.mean()))
    print("")

    # профиль вдоль ленты: в каждой полосе - число точек и максимальная высота
    n_bins = 70
    edges_s = np.linspace(s.min(), s.max(), n_bins + 1)
    print("%9s %8s %8s %8s  %s" % ("позиция", "точек", "h_max", "ширина",
                                   "профиль высоты"))
    for i in range(n_bins):
        sel = (s >= edges_s[i]) & (s < edges_s[i + 1]) & above
        count = int(sel.sum())
        if count == 0:
            continue
        h_max = (h[sel] - belt).max()
        w_span = w[sel].max() - w[sel].min()
        bar = "#" * min(60, int(h_max / 0.5))
        print("%9.1f %8d %8.1f %8.1f  %s"
              % (0.5 * (edges_s[i] + edges_s[i + 1]), count, h_max,
                 w_span, bar))

    # где самая плотная область - вероятно, там и лежит деталь
    counts = np.array([((s >= edges_s[i]) & (s < edges_s[i + 1]) & above).sum()
                       for i in range(n_bins)])
    if counts.max() > 0:
        best = int(np.argmax(counts))
        print("")
        print("Самая плотная область: около %.1f (%d точек в полосе)"
              % (0.5 * (edges_s[best] + edges_s[best + 1]), counts[best]))
        print("Полос с точками: %d из %d" % ((counts > 0).sum(), n_bins))


if __name__ == "__main__":
    main()
