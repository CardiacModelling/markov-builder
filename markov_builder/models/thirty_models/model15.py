from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_15(MarkovChain):
    description = "IKr Markov model 15 (Sanmitra Ghosh)"

    # States
    states = ('C3', 'C2', 'C1', 'O', 'I')

    # Transitions derived directly from dot() equations
    rates = [
        ('C3', 'C2', 'a1', 'b1'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C1', 'O', 'a3', 'b3'),
        ('O', 'I', 'a4', 'b4'),
    ]

    # Open state
    open_state = 'O'

    # Rate expressions (verbatim from .mmt)
    rate_dictionary = {
        'a1': ('p1 * exp(p2 * V)',),
        'b1': ('p3 * exp(-p4 * V)',),

        'a2': ('p5 * exp(p6 * V)',),
        'b2': ('p7 * exp(-p8 * V)',),

        'a3': ('p9 * exp(p10 * V)',),
        'b3': ('p11 * exp(-p12 * V)',),

        'a4': ('p13 * exp(p14 * V)',),
        'b4': ('p15 * exp(-p16 * V)',),
    }

    # Shared variables and parameters
    shared_variables_dict = {
        'V': nan,

        'p1': 2.26e-4,
        'p2': 0.06990,
        'p3': 3.45e-5,
        'p4': 0.05462,

        'p5': 0.08730,
        'p6': 8.91e-3,
        'p7': 5.15e-3,
        'p8': 0.03158,

        'p9': 2.26e-4,
        'p10': 0.06990,
        'p11': 3.45e-5,
        'p12': 0.05462,

        'p13': 0.08730,
        'p14': 8.91e-3,
        'p15': 5.15e-3,
        'p16': 0.03158,

        'p17': 0.15240,
    }

    # Current expression
    auxiliary_expression = 'p17 * state_O * (V - E_Kr)'
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {
        'E_Kr': -88,
    }

    def __init__(self):
        super().__init__(
            states=self.states,
            transition_rates=self.rates,
            rate_expressions=self.rate_dictionary,
            auxiliary_expression=self.auxiliary_expression,
            auxiliary_symbol=self.auxiliary_symbol,
            shared_variables=self.shared_variables_dict,
            auxiliary_parameters=self.auxiliary_parameters,
        )

        self.set_state_attribute('O', open_state=True)
