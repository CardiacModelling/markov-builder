from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_07(MarkovChain):
    description = ""
    states = ('O', 'C1', 'C2', 'I')
    rates = [
        ('O', 'C1', 'bm*2', 'am'),
        ('C1', 'C2', 'bm', 'am*2'),
        ('O', 'I', 'a1', 'b1')
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
                             'p9': 0.15240,
                             }

    rate_dictionary = {
        'am': ('p1 * exp(p2*V)',),
        'bm': ('p3 * exp(-p4*V)',),
        'a1': ('p5 * exp(p6*V)',),
        'b1': ('p7 * exp(-p8*V)',),
    }

    auxiliary_expression = "p9 * {} * (V - E_Kr)"
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
