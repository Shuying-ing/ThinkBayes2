"""
Today's python techniques:

- plt.bar(alpha=0.4)  # Set transparency
- make_binomial(n, p) # construct a binom distribution
"""

from empiricaldist import Pmf
import numpy as np
import pandas as pd
from utils import make_binomial
from utils import decorate
import matplotlib.pyplot as plt


def odds(p):
    """Compute a odds with the given prob."""
    return p / (1 - p)


def prob(o):
    """Compute a prob with the given odds."""
    return o / (o + 1)


def prob2(yes, no):
    """Compute a odds with the yes/no prob."""
    return yes / (yes + no)


def make_die(sides):
    "Construct the uniform distribution."
    outcomes = np.arange(1, sides + 1)
    die = Pmf(1 / sides, outcomes)
    return die


def add_dist(pmf1, pmf2):
    """Compute the distribution of a sum."""
    res = Pmf()
    for q1, p1 in pmf1.items():
        for q2, p2 in pmf2.items():
            q = q1 + q2
            p = p1 * p2
            res[q] = res(q) + p
    return res


def add_dist_seq(seq):
    """Compute Pmf of the sum of values from seq."""
    total = seq[0]
    for other in seq[1:]:
        total = total.add_dist(other)
    return total


if __name__ == "__main__":
    """
    Solve The Cookie Problem with odds | Bayes's Rule
    """
    print("\n\n------------- Solve The Cookie Problem -------------")
    prior_odds = (1 / 2) / (1 / 2)  # o(A)
    likelihood_ratio = (3 / 4) / (1 / 2)  # P(D|A)/P(D|B)
    post_odds = prior_odds * likelihood_ratio  # o(A|D)
    post_odds
    post_prob = prob(post_odds)  # Convert back to probability
    print("The probability that it came from Bowl 1 is {:.4f}".format(post_prob))

    """
    Solve The Die Problem | Addends
    Suppose you roll two dice and add them up. What is the distribution of the sum?
    """
    print("\n\n------------- Solve The Die Problem -------------")
    # Comupte the outcome of one die
    die = make_die(6)
    
    # Add the outcome of two dice
    twice = die.add_dist(die)

    # Add the outcome of three dice
    dice = [die] * 3
    thrice = add_dist_seq(dice)

    plt.figure()
    die.plot(label="once", alpha=0.4)
    twice.plot(label="twice", ls="--", alpha=0.4)
    thrice.plot(label="thrice", ls=":", alpha=0.4)
    decorate(xlabel="Outcome", ylabel="PMF", title="Distributions of sums")

    """
    Solve The forward Gluten Sensitivity Problem | The Forward Problem
    Given the number of how many subjects are sensitive, compute the distribution of the data.
    """
    # Construct the prior distribution
    n = 35
    num_sensitive = 10
    num_insensitive = n - num_sensitive
    dist_sensitive = make_binomial(num_sensitive, 0.95)
    dist_insensitive = make_binomial(num_insensitive, 0.40)

    # Compute the distribution of the total number of correct identifications
    dist_total = Pmf.add_dist(dist_sensitive, dist_insensitive)
    plt.figure()
    dist_sensitive.plot(label="sensitive", ls=":")
    dist_insensitive.plot(label="insensitive", ls="--")
    dist_total.plot(label="total")
    decorate(
        xlabel="Number of correct identifications",
        ylabel="PMF",
        title="Gluten sensitivity",
    )

    """
    Solve The Inverse Gluten Sensitivity Problem | The Inverse Problem
    Given the likelihood of the data, compute the posterior distribution of the number of sensitive patients.
    """
    # Loop through the possible values of num_sensitive and compute the distribution of the data for each
    table = pd.DataFrame()
    for num_sensitive in range(0, n + 1):
        num_insensitive = n - num_sensitive
        dist_sensitive = make_binomial(num_sensitive, 0.95)
        dist_insensitive = make_binomial(num_insensitive, 0.4)
        dist_total = Pmf.add_dist(dist_sensitive, dist_insensitive)
        table[num_sensitive] = dist_total
    table.head(3)
    plt.figure()
    table[0].plot(label="num_sensitive = 0")
    table[10].plot(label="num_sensitive = 10")
    table[20].plot(label="num_sensitive = 20", ls="--")
    table[30].plot(label="num_sensitive = 30", ls=":")
    decorate(
        xlabel="Number of correct identifications",
        ylabel="PMF",
        title="Gluten sensitivity",
    )

    # Use the above distribution to compute the likelihood of the data
    hypos = np.arange(n + 1)
    prior = Pmf(1, hypos)
    likelihood1 = table.loc[12]
    posterior1 = prior * likelihood1
    posterior1.normalize()
    likelihood2 = table.loc[20]
    posterior2 = prior * likelihood2
    posterior2.normalize()
    plt.figure()
    posterior1.plot(label="posterior with 12 correct", color="C4")
    posterior2.plot(label="posterior with 20 correct", color="C1")
    decorate(
        xlabel="Number of sensitive subjects",
        ylabel="PMF",
        title="Posterior distributions",
    )

    plt.show()
