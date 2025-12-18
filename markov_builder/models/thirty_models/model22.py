from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_22(MarkovChain):
    description = "IKr Markov model 22 (Sanmitra Ghosh)"

    # States
    states = (
        'C3', 'C2', 'C1', 'O',
        'I', 'IC1', 'IC2', 'IC3'
    )

    rates = [
        ('C3', 'C2', '3*am', 'bm'),
        ('C2', 'C1', '2*am', '2*bm'),
        ('C1', 'O', 'am', '3*bm'),

        ('O', 'I', 'a1', 'b1'),
        ('C1', 'IC1', 'a2', 'b2'),
        ('C2', 'IC2', 'a3', 'b3'),
        ('C3', 'IC3', 'a4', 'b4'),

        ('IC1', 'I', 'am', '3*bm'),
        ('IC2', 'IC1', '2*am', '2*bm'),
        ('IC3', 'IC2', '3*am', 'bm'),
    ]

    # Rate expressions
    rate_dictionary = {
        'am': ('p1 * exp(p2 * V)',),
        'bm': ('p3 * exp(-p4 * V)',),

        'a1': ('p5 * exp(p6 * V)',),
        'b1': ('p7 * exp(-p8 * V)',),

        'a2': ('p9 * exp(p10 * V)',),
        'b2': ('(a2 * b1) / a1',),

        'a3': ('p11 * exp(p12 * V)',),
        'b3': ('(a3 * b2) / a2',),

        'a4': ('p13 * exp(p14 * V)',),
        'b4': ('(a4 * b3) / a3',),
    }

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

        'p15': 0.15240,
    }

    auxiliary_expression = 'p15 * state_O * (V - E_Kr)'
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
