from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_19(MarkovChain):
    description = "IKr Markov model 19 (Sanmitra Ghosh)"

    # States
    states = ('C', 'O_1', 'I', 'O_2')

    # Transitions derived directly from dot() equations
    rates = [
        ('C', 'O_1', 'am', 'bm'),
        ('I', 'O_2', 'bh', 'ah')
    ]

    # Rate expressions (verbatim from .mmt)
    rate_dictionary = {
        'am': ('p1 * exp(p2 * V)',),
        'bm': ('p3 * exp(-p4 * V)',),
        'ah': ('p5 * exp(p6 * V)',),
        'bh': ('p7 * exp(-p8 * V)',),
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

        'p9': 0.15240,
    }

    # Current expression
    auxiliary_expression = 'p9 * state_O_1**3 * state_O_2 * (V - E_Kr)'
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {
        'E_Kr': -85,
    }

    def __init__(self):
        super().__init__(
            states=self.states,
            # open_state=self.open_state,
            transition_rates=self.rates,
            rate_expressions=self.rate_dictionary,
            auxiliary_expression=self.auxiliary_expression,
            auxiliary_symbol=self.auxiliary_symbol,
            shared_variables=self.shared_variables_dict,
            auxiliary_parameters=self.auxiliary_parameters,
        )
