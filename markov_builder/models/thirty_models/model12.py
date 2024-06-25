from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_12(MarkovChain):
    description = ""
    states = ('O', 'I', 'IC1', 'IC2', 'C2', 'C1')
    rates = [
        ('O', 'I', 'a1', 'b1'),
        ('C1', 'O', 'am', '2*bm'),
        ('IC1', 'I', 'am', '2*bm'),
        ('IC2', 'IC1', '2*am', 'bm'),
        ('C2', 'C1', '2*am', 'bm'),
        ('C2', 'IC2', 'a3', 'b3'),
        ('C1', 'IC1', 'a2', 'b2')
    ]

    open_state = 'O'
    shared_variables_dict = {'V': nan,
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
                             'p13': 0.15240
                             }

    rate_dictionary = {
        'am': ('p1 * exp( p2 * V)',),
        'bm': ('p3 * exp(-p4 * V)',),
        'a1': ('p5 * exp( p6 * V)',),
        'b1': ('p7 * exp(-p8 * V)',),
        'a2': ('p9 * exp( p10 * V)',),
        'a3': ('p11 * exp( p12 * V)',),

        'b2': ('a2 * b1 / a1',),
        'b3': ('a3 * b2 / a2',),
    }

    auxiliary_expression = "p13 * {} * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_params_dict = {'E_Kr': -85}

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         rates=self.rates,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict)
