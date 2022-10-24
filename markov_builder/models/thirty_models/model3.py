from numpy import NaN

from markov_builder.MarkovChain import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr


class model_03(MarkovChain):
    description = ""
    states = ('O', 'C', 'I', 'IC')
    rates = [('O', 'C', 'b1', 'a1'),
             ('I', 'IC', 'a3', 'b3'),
             ('O', 'I', 'a4', 'b4'),
             ('C', 'IC', 'a4', 'b4')]

    open_state = 'O'
    shared_variables_dict = {'V': NaN}
    rate_dictionary = {'a1': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'b1': negative_rate_expr + ((3.45E-5, 5.462E-2),),
                       'a3': negative_rate_expr + ((5.15e-3, 0.03158),),
                       'b3': ('(a3 * a1)/b1', (), ()),
                       'a4': positive_rate_expr + ((0.08730, 8.91e-3),),
                       'b4': negative_rate_expr + ((5.15e-3, 0.03158),)
                       }

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
