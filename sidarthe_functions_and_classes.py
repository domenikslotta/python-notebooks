from typing import List
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class Parameters8:
    """
    A class to save all default parameters for SIDARTHE calculations with 8 equations.

    Args:
        arg1    (type): description
    Attributes:
        att1    (type): description
    """

    def __init__(self,
                 alpha: float = 0.15,
                 beta: float = 0.008443847910426,
                 gamma: float = 0.15,
                 epsilon: float = 0.0,
                 zeta: float = 0.079018779922852,
                 theta: float = 0.198057106656095,
                 kappa: float = 0.056288791238505,
                 my_lambda: float = 0.059610666036892,
                 mu: float = 0.007998690003693 + 0.005027246740356,
                 sigma: float = 0.007998690003693 /
                 (0.007998690003693 + 0.005027246740356) * 0.037040181047319 +
                 0.005027246740356 /
                 (0.007998690003693 + 0.005027246740356) * 0.055231044651498,
                 tau: float = 0.007998690003693 /
                 (0.007998690003693 + 0.005027246740356) * 0.015874367480393 +
                 0.005027246740356 /
                 (0.007998690003693 + 0.005027246740356) * 0.024198833665235,
                 n_total: int = 83000000,
                 y0 = None,
                 y0_abs: List[int] = [
                     82636256, 20581, 0, 8041, 41931, 11469, 276911, 4810
                 ]):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.zeta = zeta
        self.theta = theta
        self.kappa = kappa
        self.my_lambda = my_lambda
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.n_total = n_total
        self.y0_abs = y0_abs
        if y0 is None:
            self.y0 = [x / self.n_total for x in self.y0_abs]
        else:
            self.y0 = y0

            
def sidarthe_system8(parameters: Parameters8):
    """doc"""
    alpha = parameters.alpha
    beta = parameters.beta
    gamma = parameters.gamma
    epsilon = parameters.epsilon
    theta = parameters.theta
    zeta = parameters.zeta
    my_lambda = parameters.my_lambda
    kappa = parameters.kappa
    mu = parameters.mu
    sigma = parameters.sigma
    tau = parameters.tau

    def dF(t, y) -> np.array:
        """doc"""
        return np.array([
            -y[0] * (alpha * y[1] + beta * (y[2] + y[4]) + gamma * y[3]),
            y[0] * (alpha * y[1] + beta * (y[2] + y[4]) + gamma * y[3]) -
            (epsilon + zeta + my_lambda) * y[1],
            epsilon * y[1] - (zeta + my_lambda) * y[2],
            zeta * y[1] - (theta + mu + kappa) * y[3],
            zeta * y[2] + theta * y[3] - (mu + kappa) * y[4],
            mu * (y[3] + y[4]) - (sigma + tau) * y[5],
            my_lambda * (y[1] + y[2]) + kappa * (y[3] + y[4]) + sigma * y[5],
            tau * y[5]
        ])

    return dF


def sidarthe_simulation8(parameters: Parameters8, t_0: int = 0, t_end: int = 730, steps_per_day: int = None):
    """doc"""
    dF = sidarthe_system8(parameters)

    y0 = parameters.y0
    if steps_per_day == None:
        steps_per_day = 1
    t = np.linspace(t_0, t_end, steps_per_day*(t_end-t_0) + 1)
    y = np.zeros((len(t), len(y0)))  # array for solution
    y[0, :] = y0

    r = integrate.ode(dF).set_integrator("dopri5")
    r.set_initial_value(y0, t_0)  # initial values

    for i in range(1, t.size):
        y[i, :] = r.integrate(t[i])  # get one more value, add it to the array
        if not r.successful():
            raise RuntimeError("Could not integrate")
    return t, y


def plot_all_single_8(t, y) -> None:
    """doc"""
    #plt.figure(figsize=(16, 9), dpi=160)
    plt.figure(figsize=(16, 9), dpi=80)

    plt.subplot(3, 3, 1)
    plt.plot(t, y[:, 0])
    plt.title("S")

    plt.subplot(3, 3, 2)
    plt.plot(t, y[:, 1])
    plt.title("I")

    plt.subplot(3, 3, 3)
    plt.plot(t, y[:, 2])
    plt.title("D")

    plt.subplot(3, 3, 4)
    plt.plot(t, y[:, 3])
    plt.title("A")

    plt.subplot(3, 3, 5)
    plt.plot(t, y[:, 4])
    plt.title("R")

    plt.subplot(3, 3, 6)
    plt.plot(t, y[:, 5])
    plt.title("T")

    plt.subplot(3, 3, 7)
    plt.plot(t, y[:, 6])
    plt.title("H")

    plt.subplot(3, 3, 8)
    plt.plot(t, y[:, 7])
    plt.title("E")

    plt.subplots_adjust(top=0.92,
                        bottom=0.08,
                        left=0.10,
                        right=0.95,
                        hspace=.5,
                        wspace=.5)

    plt.show()


def plot_all_compare(t, y, y_sim, text="hier könnte etwas stehen"):
    """Used to plot given data points to learn the model and compare it with simulated model"""
    #plt.figure(figsize=(16, 9), dpi=160)
    plt.figure(figsize=(16, 9), dpi=100)

    plt.subplot(3, 3, 1)
    plt.plot(t, y[:, 0], color="blue")
    plt.plot(t, y_sim[:, 0], color="red")
    plt.title("S")

    plt.subplot(3, 3, 2)
    plt.plot(t, y[:, 1], color="blue")
    plt.plot(t, y_sim[:, 1], color="red")
    plt.title("I")

    plt.subplot(3, 3, 3)
    plt.plot(t, y[:, 2], color="blue")
    plt.plot(t, y_sim[:, 2], color="red")
    plt.title("D")

    plt.subplot(3, 3, 4)
    plt.plot(t, y[:, 3], color="blue")
    plt.plot(t, y_sim[:, 3], color="red")
    plt.title("A")

    plt.subplot(3, 3, 5)
    plt.plot(t, y[:, 4], color="blue")
    plt.plot(t, y_sim[:, 4], color="red")
    plt.title("R")

    plt.subplot(3, 3, 6)
    plt.plot(t, y[:, 5], color="blue")
    plt.plot(t, y_sim[:, 5], color="red")
    plt.title("T")

    plt.subplot(3, 3, 7)
    plt.plot(t, y[:, 6], color="blue")
    plt.plot(t, y_sim[:, 6], color="red")
    plt.title("H")

    plt.subplot(3, 3, 8)
    line1, = plt.plot(t, y[:, 7], color="blue")
    line2, = plt.plot(t, y_sim[:, 7], color="red")
    plt.title("E")

    ax = plt.subplot(3, 3, 9)
    ax.axis("off")
    ax.text(0.5,
            0.45,
            text,
            size=14,
            ha="center",
            bbox={
                'facecolor': 'white',
                'linewidth': 1.5
            })

    plt.legend([line1, line2], ["exact solution", "model solution"],
               bbox_to_anchor=(0.5, -0.15),
               loc='lower center',
               fontsize=14)

    plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        left=0.05,
                        right=0.95,
                        hspace=0.5,
                        wspace=0.5)

    plt.show()