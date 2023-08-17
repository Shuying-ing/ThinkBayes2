"""
Today's python techniques:

binom.pmf(ks, n, p)  # Construct a binom distribution
print(f"xxx {variable}")  # print with 'f'
"""

from scipy.stats import binom
from empiricaldist import Pmf
import numpy as np
import matplotlib.pyplot as plt


def make_binomial(n, p):
    """Make a binomial Pmf."""
    ks = np.arange(n + 1)  # data point
    ps = binom.pmf(ks, n, p)  # prob
    return Pmf(ps, ks)


def prob_ge(pmf, threshold):
    """Probability of quantities greater than threshold."""
    ge = pmf.qs >= threshold  # <class 'numpy.ndarray'>, bool
    total = pmf[ge].sum()
    return total


def update_euro(pmf, dataset):
    """Update pmf with a given sequence of H and T."""
    for data in dataset:
        pmf *= likelihood[data]

    pmf.normalize()  # Nnormalize finally in case of floating-point arithmetic


if __name__ == "__main__":
    """
    Construct binomial distribution | The Binomial Distribution
    """
    print("\n\n------------- Construct binomial distribution -------------")
    # 01-Construct distribution from one point
    n = 2
    p = 0.5
    k = 1
    prob = binom.pmf(k, n, p)
    print(
        f"The probability with the binomial distribution given by n={n}, p={p}, k={k} is {prob}."
    )

    # 02-Construct distribution from a series of points
    pmf_k = make_binomial(n=250, p=0.5)
    plt.figure()
    pmf_k.plot(label="n=250, p=0.5")
    plt.legend()
    plt.xlabel("Number of heads (k)")
    plt.ylabel("PMF")
    plt.title("Binomial distribution")

    # Find the most likely probability
    idx = pmf_k.max_prob()  # 01
    # idx = pmf_k.idxmax()  # 02
    MAP = pmf_k[idx]
    print("\nThe maximum probability is {:.4f}, and the index is {}.".format(MAP, idx))

    # Sum up the probabilities exceeding a threshold
    threshold = 140
    prob_total1 = prob_ge(pmf_k, threshold)  # 01
    prob_total2 = pmf_k.prob_ge(threshold)  # 02
    print(
        "\nThe probability of outcomes>{} in n={} is {:.4f}.".format(
            threshold, n, prob_total1
        )
    )

    """
    Solve The Euro Problem | Bayesian Estimation
    """
    print("\n\n------------- Solve The Euro Problem -------------")
    hypos = np.linspace(0, 1, 101)  # 101 is changable
    prior = Pmf(1, hypos)  # P(H)
    likelihood_heads = hypos
    likelihood_tails = 1 - hypos
    likelihood = {"H": likelihood_heads, "T": likelihood_tails}  # P(D|H)
    dataset = "H" * 140 + "T" * 110  # D, <class 'str'>

    # Compute posteriors
    posterior = prior.copy()  # P(H|D)
    update_euro(posterior, dataset)  # Update 250 times

    # Show the proportion of heads for the coin we observed
    plt.figure()
    prior.plot(label="prior")
    posterior.plot(label="140 heads out of 250", color="C4")
    plt.legend()
    plt.xlabel("Proportion of heads (x)")
    plt.ylabel("Probability")
    plt.title("Posterior distribution of x'")

    # Find the most likely probability
    print(
        "The most likely probability of landing heads up when spun on edge is {:.4f}.".format(
            posterior.max_prob()
        )
    )

    """
    Solve The Euro Problem with swamped priors | Triangle Prior
    """
    print("\n\n------------- Solve The Euro Problem with swamped priors -------------")
    # Construct Uniform Prior
    uniform = Pmf(1, hypos, name="uniform")
    uniform.normalize()

    # Construct Triangle Prior
    ramp_up = np.arange(50)
    ramp_down = np.arange(50, -1, -1)
    a = np.append(ramp_up, ramp_down)
    triangle = Pmf(a, hypos, name="triangle")
    triangle.normalize()

    # Viasualize priors
    plt.figure()
    uniform.plot(label="uniform")
    triangle.plot(label="triangle")
    plt.legend()
    plt.xlabel("Proportion of heads (x)")
    plt.ylabel("Probability")
    plt.title("Uniform and triangle prior distributions")

    # Compute posteriors
    update_euro(uniform, dataset)
    update_euro(triangle, dataset)

    # Viasualize posteriors
    plt.figure()
    uniform.plot(label="uniform")
    triangle.plot(label="triangle")
    plt.legend()
    plt.xlabel("Proportion of heads (x)")
    plt.ylabel("Probability")
    plt.title("Posterior distributions")

    """
    [Exercise]
    Suppose you survey 100 people this way and get 80 YESes and 20 NOs.
    - Based on this data, what is the posterior distribution for the fraction of people who cheat on their taxes?
    - What is the most likely quantity in the posterior distribution?
    """
    hypos = np.linspace(0, 1, 101)
    prior = Pmf(1, hypos)  # P(H)
    likelihood = {"Y": 0.5 + hypos / 2, "N": (1 - hypos) / 2}  # P(D|H)
    dataset = "Y" * 80 + "N" * 20
    posterior = prior.copy()
    for data in dataset:
        posterior *= likelihood[data]
    posterior.normalize()

    # Viasualize posteriors
    plt.figure()
    posterior.plot(label="80 YES, 20 NO")
    plt.legend()
    plt.xlabel("Proportion of cheaters")
    plt.ylabel("PMF")
    plt.title("Posterior distributions")

    # Find the most likely probability
    print(
        "The most likely probability of people who cheat on their taxes is {:.4f}.".format(
            posterior.max_prob()
        )
    )

    plt.show()
