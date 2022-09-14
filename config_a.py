import numpy as np
import matplotlib.pyplot as plt
import math

from helper_funcs import calc_edge_lengths

# DIAMOND


def motion(eps, eta, t):
    V_to_R2 = []

    for i, j in zip(eps, eta):
        V_to_R2.append(i * math.sqrt(1 + t * (1 / i**2)))
        V_to_R2.append(j * math.sqrt(1 - t * (1 / j**2)))

    return np.array(V_to_R2).reshape(6, 1)


########################
d = 2

V_map = np.array(
    [
        [1, 0],  # 1
        [0, 1],  # 2
        [1, 0],  # 3
        [0, 1],  # 4
        [1, 0],  # 5
        [0.0, 1],  # 6 add to x to make it rigid
    ]
)

E = np.array(
    [
        [1, 6],  # 1
        [3, 6],  # 2
        [5, 6],  # 3
        [1, 4],  # 4
        [3, 4],  # 5
        [4, 5],  # 6
        [1, 2],  # 7
        [2, 3],  # 8
        [2, 5],  # 8
    ]
)

test_motion = np.array(
    [
        -1 / 12,
        0,
        0,
        -1 / 8,
        1 / 2,
        0,
        0,
        -1 / 4,
        1 / 8,
        0,
        0,
        1 / 8,
    ]
)


eps = np.array([-6, 1, 4])
eta = np.array([4, 2, -4])

V_to_R2 = motion(eps, eta, t=0)

# Original System
V = V_map * V_to_R2


##############################
def run():
    print(V)
    calc_edge_lengths(V, E)

    print("plot motion")
    fig, axs = plt.subplots(4, 4)
    fig.figure.set_size_inches(10, 10)

    t_factor = 0.25

    for t, ax in enumerate(axs.flatten()):

        print("\nplot {}, t={}".format(t, t * t_factor))

        V_to_R2 = motion(eps, eta, t * t_factor)
        print(V_to_R2)
        V_move = V_map * V_to_R2

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
        ax.set_title("t={}".format(t * t_factor))
        ax.axhline(0, color="k", linewidth=0.2)
        ax.axvline(0, color="k", linewidth=0.2)

        calc_edge_lengths(V_move, E)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
