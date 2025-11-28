from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_20(MarkovChain):
    description = ""
    states = ('O', 'O2', 'C2', 'C1', 'C3', 'I')
    rates = [
        ('C1', 'O', 'a3', 'b3'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C3', 'C2', 'a1', 'b1'),
        ('O2', 'I', 'bh', 'ah'),
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
                             'p13': 0.08730,
                             'p14': 8.91e-3,
                             'p15': 5.15e-3,
                             'p16': 0.03158,
                             'p17': 0.15240,
                             }

    rate_dictionary = {
        'a1': ('p1 * exp(p2*V)',),
        'b1': ('p3 * exp(-p4*V)',),
        'bh': ('p5 * exp(p6*V)',),
        'ah': ('p7 * exp(-p8*V)',),
        'a2': ('p9 * exp(p10*V)',),
        'b2': ('p11 * exp(-p12*V)',),
        'a3': ('p13 * exp(p14*V)',),
        'b3': ('p15 * exp(-p16*V)',)
    }

    auxiliary_expression = "p17 * state_O * state_O2 * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {'E_Kr': -85}

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         transition_rates=self.rates,
                         rate_expressions=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables=self.shared_variables_dict,
                         auxiliary_parameters=self.auxiliary_parameters
                         )
