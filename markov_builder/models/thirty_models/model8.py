from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_08(MarkovChain):
    description = ""
    states = ('O', 'Y1', 'Y2', 'Y4')
    rates = [
        ('O', 'Y2', 'k43', 'k34'),
        ('O', 'Y4', 'k56', 'k65'),
        ('Y1', 'Y2', 'k12', 'k21')
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
                             'p9': 0.08730,
                             'p10': 8.91e-3,
                             'p11': 5.15e-3,
                             'p12': 0.03158,
                             'p13': 0.15240
                             }

    rate_dictionary = {
        'k12': ('p1 * exp( p2 * V)',),
        'k21': ('p3 * exp(-p4 * V)',),
        'k34': ('p5 * exp( p6 * V)',),
        'k43': ('p7 * exp(-p8 * V)',),
        'k56': ('p9 * exp( p10 * V)',),
        'k65': ('p11 * exp(-p12 * V)',)
    }

    auxiliary_expression = "p13 * {} * (V - E_Kr)"
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
