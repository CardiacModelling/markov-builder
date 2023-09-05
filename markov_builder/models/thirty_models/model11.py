from numpy import NaN

from markov_builder.MarkovChain import MarkovChain


class model_11(MarkovChain):
    description = ""
    states = ('O', 'I', 'IC1', 'IC2', 'C2', 'C1')
    rates = [
        ('O', 'I', 'ah', 'bh'),
        ('C1', 'O', 'a1', 'b1'),
        ('IC1', 'I', 'a3', 'b3'),
        ('IC2', 'IC1', 'a4', 'b4'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C2', 'IC2', 'ah', 'bh'),
        ('C1', 'IC1', 'ah', 'bh')
    ]

    open_state = 'O'
    shared_variables_dict = {'V': NaN,
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
                             'p17': 0.15240
                             }

    rate_dictionary = {
        'a1': ('p1 * exp( p2 * V)',),
        'b1': ('p3 * exp(-p4 * V)',),
        'ah': ('p5 * exp( p6 * V)',),
        'bh': ('p7 * exp(-p8 * V)',),
        'a2': ('p9 * exp( p10 * V)',),
        'b2': ('p11 * exp(-p12 * V)',),
        'a3': ('p13 * exp( p14 * V)',),
        'b3': ('(a3*b1)/a1',),
        'a4': ('p15 * exp(p16* V)',),
        'b4': ('(a4*b2)/a2',)
    }

    auxiliary_expression = "p17 * state_O * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_params_dict = {'E_Kr': -88}

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         rates=self.rates,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict)
