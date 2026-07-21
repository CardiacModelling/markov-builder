from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_14(MarkovChain):
    description = ""
    states = ('O', 'C2', 'C1', 'C3', 'I')
    rates = [
        ('C1', 'O', 'am', '3*bm'),
        ('C2', 'C1', '2*am', '2*bm'),
        ('C3', 'C2', '3*am', 'bm'),
        ('O', 'I', 'b1', 'a1'),
    ]

    shared_variables_dict = {'V': nan,
                             'p1': 2.26e-4,
                             'p2': 0.06990,
                             'p3': 3.45e-5,
                             'p4': 0.05462,
                             'p5': 0.08730,
                             'p6': 8.91e-3,
                             'p7': 5.15e-3,
                             'p8': 0.03158,
                             'p9': 0.1524
                             }

    rate_dictionary = {
        'am': ('p1 * exp( p2 * V)',),
        'bm': ('p3 * exp(-p4 * V)',),
        'b1': ('p5 * exp(p6 * V)',),
        'a1': ('p7 * exp(-p8 * V)',),
    }

    auxiliary_expression = "p9 * state_O * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {'E_Kr': -85}

    def __init__(self):
        super().__init__(states=self.states,
                         transition_rates=self.rates,
                         rate_expressions=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables=self.shared_variables_dict,
                         auxiliary_parameters=self.auxiliary_parameters
                         )

        self.set_state_attribute('O', open_state=True)
