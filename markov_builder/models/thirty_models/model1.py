from numpy import NaN

from markov_builder.MarkovChain import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr


class model_01(MarkovChain):
    description = ""
    states = ('O', 'C')
    rates = [('O', 'C', 'k_21', 'k_12')]

    shared_variables_dict = {
        'V': NaN,
        'g_Kr': 0.1524,
    }

    rate_dictionary = {'k_12': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'k_21': negative_rate_expr + ((3.45e-5, 0.05462),)}

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
