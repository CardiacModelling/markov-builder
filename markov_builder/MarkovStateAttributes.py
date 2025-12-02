from dataclasses import dataclass


@dataclass
class MarkovStateAttributes:
    """A dataclass defining what attributes each state in a MarkovChain should have, and their default values.

    Additional attributes may be included by specifying another dataclass
    """

    open_state: bool = False
    drug_bound: bool = False
