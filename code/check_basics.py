"""공책에 있는 예제를 다시 실행한다.

경사하강, 원소별 덧셈과 파이썬 루프, 브로드캐스팅, 행렬곱, reshape.
그림이 필요하면 --save 를 붙인다. 저장 위치는 results/ 다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def naive_add(x, y):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = x[i, j] + y[i, j]
    return out


def gradient_descent_simple(lr=0.1, epochs=50):
    x = 5.0
    history = [x]
    for _ in range(epochs):
        grad = 2 * x
        x = x - lr * grad
        history.append(x)
    return x, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    a = rng.random((400, 400)).astype("float32")
    b = rng.random((400, 400)).astype("float32")

    t0 = time.perf_counter()
    z_fast = a + b
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_slow = naive_add(a, b)
    t_slow = time.perf_counter() - t0
    print(f"x + y      {t_fast:.6f} s")
    print(f"naive_add  {t_slow:.6f} s")
    print("allclose  ", bool(np.allclose(z_fast, z_slow)))

    x = rng.random((32, 10))
    y = rng.random((10,))
    print("broadcast ", (x + y).shape)
    x = rng.random((32, 10))
    y = rng.random((10, 4))
    print("dot       ", (x @ y).shape)

    grid = np.array([[0, 1], [2, 3], [4, 5]])
    print("reshape 6,1\n", grid.reshape((6, 1)))
    print("reshape 2,3\n", grid.reshape((2, 3)))

    final_x, hist = gradient_descent_simple()
    print("gd final  ", f"{final_x:.8e}")
    print("gd head   ", [round(v, 4) for v in hist[:5]])

    if not args.save:
        return

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(-1, 6, 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(xs, xs ** 2, color="black", label="y = x^2")
    ax.plot(hist, np.square(hist), "o-", color="#1f4e79", ms=3, label="GD")
    ax.scatter([0], [0], color="#a33b32", zorder=3)
    ax.set_title(f"final x = {final_x:.3e}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gradient-x2.png", dpi=140)
    print("saved", OUT / "gradient-x2.png")


if __name__ == "__main__":
    main()
