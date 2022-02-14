from dataclasses import dataclass


@dataclass
class MarkovStateAttributes:
    """A dataclass defining what attributes each state in a MarkovChain should have, and their default values

    In order to include additional attributes, you should create a subclass of
    this class. Then you can create a MarkovChain with the keyword argument
    state_attribute_class=my_attributes_subclass

    """
    open_state: bool = False
