"""
Today's python techniques:

- Pmf()  #  Construct distributions
- Pmf.from_seq()
- Pmf(1, hypos)
- prior.normalize()  # normalize distributions
- idx = posterior.max_prob()  # 01 Find the most likely value
- idx = posterior.idxmax()  # 02 Find the most likely value
- MAP = posterior[idx]  # Find the maximum probabiliry
"""

from empiricaldist import Pmf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def update_dice(pmf, data):
    """Update pmf based on new data."""
    hypos = pmf.qs
    likelihood = 1 / hypos  # P(D|H)
    impossible = data > hypos
    likelihood[impossible] = 0
    pmf *= likelihood
    pmf.normalize()  # P(H|D)


if __name__ == "__main__":
    """
    Construct distributions with Pmf() | Probability Mass Functions
    """
    print("\n\n------------- Construct distributions -------------")
    # Example1 -numbers quantities
    coin = Pmf()
    coin["heads"] = 1 / 2
    coin["tails"] = 1 / 2
    print("Coin:\n", coin)

    # Example2 -numbers quantities
    die = Pmf.from_seq([1, 2, 3, 4, 5, 6])
    print("\nDie:\n", die)

    # Example3 -strings quantities
    letters = Pmf.from_seq(list("Mississippi"))
    print("\nLetters:\n", letters)
    print("\nThe probability of a quantity that's not in the distribution:")
    try:
        letters["t"]
    except KeyError as e:
        print("{} (with ['t'])".format(type(e)))  # <class 'KeyError'>
    print("{} (with ('t'))".format(letters("t")))  # 0

    """
    Solve The Cookie Problem with with 101 bowls | 101 Bowls
    """
    print("\n\n------------- Solve The Cookie Problem -------------")
    hypos = np.arange(101)
    prior = Pmf(1, hypos)  # 1-uniform prior, hypos-a sequence of quantities
    prior.normalize()  # P(H)
    likelihood_vanilla = hypos / 100  # P(D|H)
    likelihood_vanilla[:5]
    likelihood_chocolate = 1 - likelihood_vanilla

    # Case1 -get a vanilla cookie from a bowl
    posterior1 = prior * likelihood_vanilla
    posterior1.normalize()  # P(H|D)
    plt.figure()
    prior.plot(label="prior", color="C5")
    posterior1.plot(label="posterior", color="C4")
    plt.legend()
    plt.xlabel("Bowl #")
    plt.ylabel("PMF")
    plt.title("Posterior after 1 vanilla cookie")

    # Case2 -put the cookie back, draw again from the same bowl, and get another vanilla cookie
    posterior2 = posterior1 * likelihood_vanilla
    posterior2.normalize()  # P(H|D)
    plt.figure()
    posterior2.plot(label="posterior", color="C4")
    plt.legend()
    plt.xlabel("Bowl #")
    plt.ylabel("PMF")
    plt.title("Posterior after 2 vanilla cookies")

    # Case3 -put the second cookie back, draw again and get a chocolate cookie
    posterior3 = posterior2 * likelihood_chocolate
    posterior3.normalize()  # P(H|D)
    plt.figure()
    posterior3.plot(label="posterior", color="C4")
    plt.legend()
    plt.xlabel("Bowl #")
    plt.ylabel("PMF")
    plt.title("Posterior after 2 vanilla, 1 chocolate")

    # Find the most likely probability
    idx = posterior3.max_prob()  # 01
    # idx = posterior3.idxmax() # 02
    MAP = posterior3[idx]
    print(
        "The maximum posteriori probability is {:.4f}, and the index is {}.".format(
            MAP, idx
        )
    )

    """
    Solve The Dice Problem (Update) | The Dice Problem
    """
    print("\n\n------------- Solve The Dice Problem (Update) -------------")
    hypos = [6, 8, 12]
    prior = Pmf(1 / 3, hypos)  # P(H)
    print("The quantities in the distribution: {}".format(prior.qs))
    print("The corresponding probabilities: {}".format(prior.ps))

    # Case1 -roll the die and get a 1
    posterior1 = prior.copy()
    update_dice(posterior1, 1)
    print("\nPosterior after 1 roll:\n", posterior1)

    # Case2 -roll the same die again and get a 7
    posterior2 = posterior1.copy()
    update_dice(posterior2, 7)
    print("\nPosterior after 2 rolls:\n{}".format(posterior2))

    """
    [Exercise]
    Elvis Presley had a twin brother (who died at birth). What is the probability that Elvis was an identical twin?
    Hint: In 1935, about 2/3 of twins were fraternal and 1/3 were identical.
    """
    print("\n\n------------- [Exercise] -------------")
    # Method1 -with Bayes Table
    table = pd.DataFrame(index=["identical", "fraternal"])
    table["prior"] = 1 / 3, 2 / 3  # P(H)
    table["likelihood"] = 1, 1 / 2  # P(D|H)
    table["unorm"] = table["prior"] * table["likelihood"]  # unnormalized P(H|D)
    prob_data = table["unorm"].sum()  # P(D)
    table["posterior"] = table["unorm"] / prob_data  # P(H|D)
    print("Posterior with Bayes Table:\n", table)

    # Method2 -with Pmf()
    hypos = ["identical", "fraternal"]
    prior = Pmf([1 / 3, 2 / 3], hypos)  # P(H)
    likelihood = 1, 1 / 2  # P(D|H)
    posterior = prior * likelihood  # unnormalized P(H|D)
    posterior.normalize()  # P(H|D)
    print("\nPosterior with Pmf:\n", posterior)

    plt.show()
