"""
Today's python techniques:

- mean = np.sum(posterior.ps * posterior.qs)    # Expectation of a distribution
"""

from empiricaldist import Pmf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Solve The Train Problem | The Train Problem
    """
    print("\n\n------------- Solve The Train Problem -------------")
    data = 60  # D
    hypos = np.arange(1, 1001)  # H
    prior = Pmf(1, hypos)  # p(H)
    likelihood = 1 / hypos
    impossible = data > hypos  # shape=(1000,)
    likelihood[impossible] = 0  # P(D|H)
    posterior = prior * likelihood  # unnormalized P(H|D)
    posterior.normalize()  # P(H|D)

    # Viasualize posteriors
    plt.figure()
    posterior.plot(label="Posterior after train 60")
    plt.legend()
    plt.xlabel("Number of trains")
    plt.ylabel("PMF")
    plt.title("Posterior distributions")

    # Find the most likely value
    print(
        "The most likely value of locomotives the railroad has is {:.2f}.".format(
            posterior.max_prob()
        )
    )

    # Compute the mean of the distribution as the most likely value
    mean = np.sum(posterior.ps * posterior.qs)  # 01
    # mean = posterior.mean  # 02
    print(
        "From the expextation perspective, the most likely value of locomotives the railroad has is {:.2f}.".format(
            mean
        )
    )

    plt.show()
