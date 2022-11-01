from numpy import NaN

from markov_builder.MarkovChain import MarkovChain


class model_05(MarkovChain):
    description = ""
    states = ('O', 'C', 'I', 'IC')
    rates = [('O', 'C', 'bm', 'am'),
             ('I', 'IC', 'bm', 'am'),
             ('O', 'I', 'b1', 'a1'),
             ('C', 'IC', 'b2', 'a2')]

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
                             'p9': 5.15e-3,
                             'p10': 0.03158,
                             'p11': 0.15240,
                             }

    rate_dictionary = {'am': ('p1 * exp(p2*V)',),
                       'bm': ('p3 * exp(-p4*V)',),
                       'b1': ('p5 * exp(p6*V)',),
                       'a1': ('p7 * exp(-p8*V)',),
                       'b2': ('p9 * exp(p10*V)',),
                       'a2': ('(b2*a1)/b1',)
                       }

    auxiliary_expression = "p11 * {} * (V - E_Kr)"
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
