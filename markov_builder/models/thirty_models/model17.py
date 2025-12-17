from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_17(MarkovChain):
    description = "IKr Markov model (Sanmitra Ghosh)"

    # States
    states = ('C3', 'C2', 'C1', 'O', 'I')

    # Transitions inferred directly from dot() equations
    rates = [
        # C3 <-> C2
        ('C3', 'C2', '3*am', 'bm'),

        # C2 <-> C1
        ('C2', 'C1', '2*am', '2*bm'),

        # C1 <-> O
        ('C1', 'O', 'am', '3*bm'),

        # C1 <-> I
        ('C1', 'I', 'a2', 'b2'),

        # O <-> I
        ('O', 'I', 'a1', 'b1'),
    ]

    # Open state
    open_state = 'O'

    # Rate expressions (verbatim from .mmt)
    rate_dictionary = {
        'am': ('p1 * exp(p2 * V)',),
        'bm': ('p3 * exp(-p4 * V)',),
        'a1': ('p5 * exp(p6 * V)',),
        'b1': ('p7 * exp(-p8 * V)',),
        'a2': ('p9 * exp(p10 * V)',),
        'b2': ('(a2 * 3 * bm * b1) / (am * a1)',),
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
        'p9': 5.15e-3,
        'p10': 0.03158,
        'p11': 0.15240,
    }

    # Current expression
    auxiliary_expression = 'p11 * state_O * (V - E_Kr)'
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {
        'E_Kr': -85,
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
