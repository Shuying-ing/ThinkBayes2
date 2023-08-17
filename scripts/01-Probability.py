"""
Today's python techniques:

print("xxx {:.4f}".format(variable))  # formatting print
selected = A[B]  # Select A given B
"""

import pandas as pd


def prob(A):
    """Computes the probability of a proposition, A."""
    return A.mean()


def conditional(proposition, given):
    """Probability of A conditioned on given."""
    return prob(proposition[given])


if __name__ == "__main__":
    """
    Prepare data
    """
    print("\n\n------------- Prepare data -------------")
    # Load GSS data
    gss = pd.read_csv("./data/gss_bayes.csv", index_col=0)
    gss.head()

    # Select data interested
    banker = gss["indus10"] == 6870  # <class 'pandas.core.series.Series'>,  dtype: bool
    banker.head()
    liberal = gss["polviews"] <= 3
    democrat = gss["partyid"] <= 1
    male = gss["sex"] == 1
    female = gss["sex"] == 2
    B = gss["polviews"]

    """
    Compute probability of people interested | Probability, Conjunction
    """
    print("\n\n------------- Compute probability of people interested -------------")
    print("The number of bankers is {}.".format(banker.sum()))
    print("The probability of bankers is {:.4f}.".format(prob(banker)))
    print("\nThe number of Democrat bankers is {}.".format((banker & democrat).sum()))
    print(
        "The probability of Democrat bankers is {:.4f}.".format(
            prob((banker & democrat))
        )
    )

    """
    Compute probability with conditions | Conditional Probability
    “Of all the respondents who are liberal, what fraction are Democrats?”
    """
    print("\n\n------------- Compute probability with conditions -------------")
    selected = democrat[liberal]
    selected.head()
    p1 = conditional(democrat, given=liberal)
    p2 = prob(democrat & liberal) / prob(liberal)
    print(
        "The probability that a respondent is a democrat given that they are liberal:"
    )
    print("{:.4f} (compute with conditional probability)".format(p1))
    print("{:.4f} (compute with conjunction)".format(p2))

    """
    Compute total probability | The Law of Total Probability
    """
    print("\n\n------------- Compute total probability -------------")
    p1 = prob(banker)
    p2 = prob(male) * conditional(banker, given=male) + prob(female) * conditional(
        banker, given=female
    )
    p3 = prob(male & banker) + prob(female & banker)
    p4 = sum(prob(B == i) * conditional(banker, B == i) for i in range(1, 8))
    print("The total probability of banker:")
    print("{:.4f} (compute directly)".format(p1))
    print("{:.4f} (compute with conditional probability, gender)".format(p2))
    print("{:.4f} (compute with total probability)".format(p3))
    print("{:.4f} (compute with conditional probability, polviews)".format(p4))
