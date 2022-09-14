import numpy as np
import matplotlib.pyplot as plt
import math

from helper_funcs import calc_edge_lengths

# DESAURGES


def motion(V, x, y, t):

    m = np.array(
        [
            x[0],
            y[0],
            x[0] + x[1],
            y[0],
            x[0] + x[1] / 2,
            y[2],
            (x[0]) + 4 * math.sin(math.radians(t)),
            y[0] + 4 * math.cos(math.radians(t)),
            (x[1]) + 4 * math.sin(math.radians(t)),
            y[0] + 4 * math.cos(math.radians(t)),
            (x[0] + x[1] / 2) + 4 * math.sin(math.radians(t)),
            y[2] + 4 * math.cos(math.radians(t)),
        ]
    )

    V_new = m.reshape(6, 2)

    return V_new


########################
d = 2

x = [0, 3]
y = [0, 4, 1]

V = np.array(
    [
        [x[0], y[0]],  # 1
        [x[0] + x[1], y[0]],  # 2
        [x[0] + x[1] / 2, y[2]],  # 3
        [x[0], y[0] + y[1]],  # 4
        [x[0] + x[1], y[0] + y[1]],  # 5
        [0.1 + x[0] + x[1] / 2, y[0] + y[1] + y[2]],  # 6 #add to x to make rigid
    ]
)

E = np.array(
    [
        [1, 2],  # 1
        [1, 3],  # 2
        [2, 3],  # 3
        [1, 4],  # 4
        [2, 5],  # 5
        [4, 5],  # 6
        [4, 6],  # 7
        [5, 6],  # 8
        [3, 6],  # 9
    ]
)

# shearing
test_motion = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        math.cos(0),
        -math.sin(0),
        math.cos(0),
        -math.sin(0),
        math.cos(0),
        -math.sin(0),
    ]
)


##############################
def run():
    print(V)
    calc_edge_lengths(V, E)

    print("plot motion")
    fig, axs = plt.subplots(4, 4)
    fig.figure.set_size_inches(10, 10)

    t_factor = 2

    for t, ax in enumerate(axs.flatten()):

        print("\nplot {}, t={}".format(t, t * t_factor))

        V_move = motion(V, x, y, t * t_factor)

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
