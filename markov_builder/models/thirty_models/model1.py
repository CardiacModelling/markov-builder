from numpy import NaN

from markov_builder.MarkovChain import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr


class model_01(MarkovChain):
    description = ""
    states = ('O', 'C')
    rates = [('O', 'C', 'k_21', 'k_12')]

    rate_dictionary = {
        'k_12': ('p1 * exp(p2*V)',),
        'k_21': ('p3 * exp(-p4*V)',)
    }

    shared_variables_dict = {'V': NaN,
                             'p1': 2.26e-4,
                             'p2': 0.06990,
                             'p3': 3.45e-5,
                             'p4': 0.05462,
                             'g_Kr': 0.1524,
                             }

    auxiliary_expression = "g_Kr * state_O * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'
    auxiliary_params_dict = {'E_Kr': -88}

    open_state = 'O'

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         rates=self.rates,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict
                         )
