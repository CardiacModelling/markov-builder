from numpy import NaN

from markov_builder.MarkovChain import MarkovChain


class model_03(MarkovChain):
    description = ""
    states = ('O1', 'C', 'O2', 'I')
    rates = [('O1', 'C', 'bm', 'am'),
             ('O2', 'I', 'ah', 'bh')]

    rate_dictionary = {'am': ('p1 * exp(V * p2)',),
                       'bm': ('p3 * exp(-V * p4)',),
                       'ah': ('p5 * exp(V * p6)',),
                       'bh': ('p7 * exp(-V * p8)',),
                       }

    shared_variables_dict = {'V': NaN,
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

    auxiliary_expression = "p9 * state_O1 * state_O2 * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'
    auxiliary_params_dict = {'E_Kr': -88}
    open_state = 'O1'

    def __init__(self):
        super().__init__(states=self.states,
                         rates=self.rates,
                         open_state=self.open_state,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict)
