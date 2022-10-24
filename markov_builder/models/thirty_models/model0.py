import numpy as np

from markov_builder import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr


class model_00(MarkovChain):
    description = ""
    states = ('O', 'C', 'I', 'IC')
    rates = [('O', 'C', 'k_2', 'k_1'),
             ('I', 'IC', 'k_2', 'k_1'),
             ('O', 'I', 'k_3', 'k_4'),
             ('C', 'IC', 'k_3', 'k_4')]

    open_state = 'O'
    shared_variables_dict = {'V': np.NaN}

    rate_dictionary = {'k_1': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'k_2': negative_rate_expr + ((3.44E-5, 5.460E-2),),
                       'k_3': positive_rate_expr + ((0.0873, 8.91E-3),),
                       'k_4': negative_rate_expr + ((5.15E-3, 0.003158),)}
    auxiliary_expression = "g_Kr * {} * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_params_dict = {'g_Kr': 0.1524,
                             'E_Kr': -88
                             }

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         rates=self.rates,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict)
