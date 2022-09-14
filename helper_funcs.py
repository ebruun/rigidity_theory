import numpy as np


def calc_edge_lengths(V, E):
    lens = []
    for e in E:
        lens.append(np.linalg.norm(V[e[0] - 1, :] - V[e[1] - 1, :]))
    print("lengths:", lens)


def plot_2d(ax, E, V, V_move):
    for i in E:
        s = i[0] - 1
        e = i[1] - 1
        ax.plot(
            [V[s][0], V[e][0]],
            [V[s][1], V[e][1]],
            "k-",
            linewidth=1,
            marker=".",
            markersize=5,
        )

    for i in E:
        s = i[0] - 1
        e = i[1] - 1
        ax.plot(
            [V_move[s][0], V_move[e][0]],
            [V_move[s][1], V_move[e][1]],
            "r-",
            linewidth=0.75,
            marker=".",
            markersize=3,
        )

    ax.axis("equal")
    ax.axhline(0, color="k", linewidth=0.2)
    ax.axvline(0, color="k", linewidth=0.2)

    for i in range(len(V)):
        ax.text(V[i, 0], V[i, 1], i + 1)
        ax.text(V_move[i, 0], V_move[i, 1], i + 1, color="red")


def plot_3d(ax, E, V, V_move):
    for i in E:
        s = i[0] - 1
        e = i[1] - 1
        ax.plot3D(
            [V[s][0], V[e][0]],
            [V[s][1], V[e][1]],
            [V[s][2], V[e][2]],
            "k-",
            linewidth=1,
            marker=".",
            markersize=5,
        )

    for i in E:
        s = i[0] - 1
        e = i[1] - 1
        ax.plot3D(
            [V_move[s][0], V_move[e][0]],
            [V_move[s][1], V_move[e][1]],
            [V_move[s][2], V_move[e][2]],
            "r-",
            linewidth=0.75,
            marker=".",
            markersize=3,
        )

    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    for i in range(len(V)):
        ax.text(V[i, 0], V[i, 1], V[i, 2], i + 1, "x")
        ax.text(V_move[i, 0], V_move[i, 1], V_move[i, 2], i + 1, "x", color="red")
