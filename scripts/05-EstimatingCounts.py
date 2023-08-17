"""
Today's python techniques:

mean = np.sum(posterior.ps * posterior.qs)  # Expectation of a distribution
df = pd.DataFrame(columns=["Posterior mean"])  # Constuct a df with columns
df.index.name = "Upper bound"  # Designate the index name of df
posterior.prob_le() # Compute percentile rank
posterior.credible_interval()  # Compute credible_interval
"""

from empiricaldist import Pmf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def update_train(pmf, data):
    """Update pmf based on new data."""
    hypos = pmf.qs
    likelihood = 1 / hypos
    impossible = data > hypos
    likelihood[impossible] = 0
    pmf *= likelihood
    pmf.normalize()


def quantile(pmf, prob):
    """Compute a quantile with the given prob."""
    total = 0
    for q, p in pmf.items():
        total += p
        if total >= prob:
            return q
    return np.nan


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

    # Compute the posterior with varied upper bound
    df = pd.DataFrame(columns=["Posterior mean"])
    df.index.name = "Upper bound"
    for N in [501, 1001, 2001]:
        hypos = np.arange(1, N)  # H
        pmf = Pmf(1, hypos)
        update_train(pmf, data=60)
        df.loc[N] = pmf.mean()
    print("With uniform prior, the posterior with varied upper bound:\n", df)

    """
    Solve The Train Problem with Power Law Prior | Power Law Prior
    """
    print(
        "\n\n------------- Solve The Train Problem with Power Law Prior -------------"
    )
    # Construct a power law prior
    hypos = np.arange(1, 1001)
    alpha = 1.0
    ps = hypos**alpha
    power = Pmf(ps, hypos, name="power law")
    power.normalize()  # p(H)

    # Construct a uniform prior as comparison
    hypos = np.arange(1, 1001)
    uniform = Pmf(1, hypos, name="uniform")
    uniform.normalize()  # p(H)

    # Viasualize priors
    plt.figure()
    power.plot(label="power law")
    uniform.plot(label="uniform")
    plt.legend()
    plt.xlabel("Number of trains")
    plt.ylabel("PMF")
    plt.title("Prior distributions")

    # Compute the posterior
    update_train(power, data=60)
    update_train(uniform, data=60)

    # Viasualize posteriors
    plt.figure()
    power.plot(label="power law")
    uniform.plot(label="uniform")
    plt.legend()
    plt.xlabel("Number of trains")
    plt.ylabel("PMF")
    plt.title("Posterior distributions")

    # Find the most likely value
    print(
        "The most likely value of locomotives the railroad has is {:.2f}.".format(
            power.max_prob()
        )
    )

    # Compute the posterior with varied upper bound
    df = pd.DataFrame(columns=["Posterior mean"])
    df.index.name = "Upper bound"
    for N in [501, 1001, 2001]:
        hypos = np.arange(1, N + 1)
        ps = hypos ** (-alpha)
        power = Pmf(ps, hypos)
        update_train(power, data)
        df.loc[N] = power.mean()
    print("With power law prior, the posterior with varied upper bound:\n", df)

    """
    Compute percentile rank and quantile | Credible Intervals
    """
    print("\n\n------------- Compute percentile rank and quantile -------------")
    # Compute percentile rank
    n = 100
    percentile = power.prob_le(n)
    print(
        "The probability that the company has less than or equal to {} trains is {:.4f}.".format(
            n, percentile
        )
    )

    # Compute quantile
    prob = 0.5
    q = quantile(power, prob)
    print(
        "The probability is {} that the number of trains is less than or equal to {}.".format(
            prob, q
        )
    )

    # Compute credible interval
    interval = power.credible_interval(0.9)
    print("The interval that contains the given probability is {}.".format(interval))

    plt.show()
