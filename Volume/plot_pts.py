"""
Визуализация .pts файла: помогает понять, где лента и как выглядят сечения.
Сохраняет 3 картинки рядом с файлом:
    <файл>_hist.png   - гистограмма высот (искать полку ленты)
    <файл>_side.png   - вид сбоку (s вдоль движения, h высота)
    <файл>_slices.png - несколько поперечных срезов (w, h)

Использование:
    python plot_pts.py файл.pts
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRECTION   = [1.0, 0.0, 0.0]
BELT_NORMAL = [0.0, 0.0, 1.0]


def main():
    path = sys.argv[1]
    data = np.loadtxt(path)
    pts = data[:, 1:4]
    stem = path.rsplit(".", 1)[0]

    u = np.asarray(DIRECTION, float); u /= np.linalg.norm(u)
    n = np.asarray(BELT_NORMAL, float); n /= np.linalg.norm(n)
    v = np.cross(n, u); v /= np.linalg.norm(v)

    s = pts @ u
    w = pts @ v
    h = pts @ n

    # гистограмма высот
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(h, bins=400)
    ax.set_yscale("log")
    ax.set_xlabel("высота (проекция на BELT_NORMAL)")
    ax.set_ylabel("число точек (log)")
    ax.set_title("Гистограмма высот: лента = узкий высокий пик снизу")
    fig.tight_layout()
    fig.savefig(stem + "_hist.png", dpi=130)

    # вид сбоку (прореживаю до ~200 тыс. точек)
    step = max(len(pts) // 200_000, 1)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(s[::step], h[::step], s=0.3, linewidths=0)
    ax.set_xlabel("s (вдоль движения)")
    ax.set_ylabel("h (высота)")
    ax.set_title("Вид сбоку")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(stem + "_side.png", dpi=130)

    # поперечные срезы в 5 местах по длине
    qs = np.percentile(s, [10, 30, 50, 70, 90])
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
    half = 0.5   # полутолщина среза
    for ax, s0 in zip(axes, qs):
        m = np.abs(s - s0) < half
        ax.scatter(w[m], h[m], s=1.0, linewidths=0)
        ax.set_title("s ~ %.1f (%d точек)" % (s0, int(m.sum())))
        ax.set_xlabel("w (поперёк)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("h (высота)")
    fig.suptitle("Поперечные срезы")
    fig.tight_layout()
    fig.savefig(stem + "_slices.png", dpi=130)

    print("сохранено:")
    for suf in ("_hist.png", "_side.png", "_slices.png"):
        print(" ", stem + suf)


if __name__ == "__main__":
    main()
