from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_02(MarkovChain):
    description = ""
    states = ('O', 'C', 'I')
    rates = [('O', 'C', 'bm', 'am'),
             ('O', 'I', 'ah', 'bh')]

    rate_dictionary = {'am': ('p1 * exp(p2 * V)',),
                       'bm': ('p3 * exp(-p4 * V)',),
                       'ah': ('p5 * exp(p6 * V)',),
                       'bh': ('p7 * exp(-p8 * V)',)
                       }

    shared_variables_dict = {'V': nan,
                             'p1': 2.26e-4,
                             'p2': 0.06990,
                             'p3': 3.45e-5,
                             'p4': 0.05462,
                             'p5': 0.08730,
                             'p6': 8.91e-3,
                             'p7': 5.15e-3,
                             'p8': 0.03158,
                             'g_Kr': 0.1524,
                             }

    auxiliary_expression = "g_Kr * state_O * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'
    auxiliary_parameters = {'E_Kr': -88}

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
