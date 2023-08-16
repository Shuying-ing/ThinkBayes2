"""
Today's python techniques:

- table = pd.DataFrame(index=["H1", "H2"])
- Fraction(num, den)
"""

import pandas as pd
from fractions import Fraction


def update(table):
    """Compute the posterior probabilities."""
    table["unnorm"] = (
        table["prior"] * table["likelihood"]
    )  # unnormalized posteriors p(H|D)
    prob_data = table["unnorm"].sum()
    table["posterior"] = table["unnorm"] / prob_data  # p(H|D)
    return prob_data


if __name__ == "__main__":

    """
    Construct a general Bayes table
    """
    print("\n\n------------- Construct a general Bayes table -------------")
    table = pd.DataFrame(index=["H1", "H2"])
    table["prior"] = 0, 0  # p(H)
    table["likelihood"] = 0, 0  # p(D|H)
    update(table)
    table.rename(
        columns={
            "prior": "prior-p(H)",
            "likelihood": "likelihood-p(D|H)",
            "unnorm": "unnormalized posteriors",
            "posterior": "posterior-p(H|D)",
        },
        inplace=True,
    )
    print("Bayes Table:\n", table)

    """
    Solve The Cookie Problem with BayesTable | Bayes Table
    """
    print("\n\n------------- Solve The Cookie Problem -------------")
    table1 = pd.DataFrame(index=["Bowl 1", "Bowl 2"])
    table1["prior"] = 1 / 2, 1 / 2  # p(H)
    table1["likelihood"] = 3 / 4, 1 / 2  # p(D|H)
    update(table1)
    print("Bayes Table for the Cookie Problem:\n", table1)

    """
    Solve The Dice Problem with BayesTable | The Dice Problem
    """
    print("\n\n------------- Solve The Dice Problem -------------")
    table2 = pd.DataFrame(index=[6, 8, 12])
    table2["prior"] = Fraction(1, 3)  # p(H)
    table2["likelihood"] = Fraction(1, 6), Fraction(1, 8), Fraction(1, 12)  # p(D|H)
    update(table2)
    print("Bayes Table for the Dice Problem:\n", table2)

    """
    Solve The Monty Hall Problem with BayesTable | The Monty Hall Problem
    """
    print("\n\n------------- Solve The Monty Hall Problem -------------")
    table3 = pd.DataFrame(index=["Door 1", "Door 2", "Door 3"])
    table3["prior"] = Fraction(1, 3)  # p(H)
    table3["likelihood"] = Fraction(1, 2), 1, 0  # p(D|H)
    update(table3)
    print("Bayes Table for the Monty Hall Problem:\n", table3)
