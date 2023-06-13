from numpy import NaN

from markov_builder.MarkovChain import MarkovChain


class model_13(MarkovChain):
    description = ""
    states = ('O', 'I', 'IC1', 'IC2', 'C2', 'C1')
    rates = [
        ('O', 'I', 'a3', 'b3'),
        ('C1', 'O', 'a2', 'b2'),
        ('IC1', 'I', 'a7', 'b7'),
        ('IC2', 'IC1', 'a6', 'b6'),
        ('C2', 'C1', 'a1', 'b1'),
        ('C2', 'IC2', 'a5', 'b5'),
        ('C1', 'IC1', 'a4', 'b4')
    ]

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
                             'p9': 2.26e-4,
                             'p10': 0.06990,
                             'p11': 3.45e-5,
                             'p12': 0.05462,
                             'p13': 2.26e-4,
                             'p14': 0.06990,
                             'p15': 3.45e-5,
                             'p16': 0.05462,
                             'p17': 0.08730,
                             'p18': 8.91e-3,
                             'p19': 5.15e-3,
                             'p20': 0.03158,
                             'p21': 2.26e-4,
                             'p22': 0.06990,
                             'p23': 3.45e-5,
                             'p24': 0.05462,
                             'p25': 0.15240,
                             }

    rate_dictionary = {
        'a1': ('p1 * exp( p2 * V)',),
        'b1': ('p3 * exp(-p4 * V)',),
        'a2': ('p5 * exp( p6 * V)',),
        'b2': ('p7 * exp(-p8 * V)',),
        'a3': ('p9 * exp( p10 * V)',),
        'b3': ('p11 * exp(-p12 * V)',),
        'a4': ('p13 * exp( p14 * V)',),
        'b4': ('(a7*b3*b2*a4)/(a2*a3*b7)',),
        'a5': ('p15 * exp( p16 * V)',),
        'b5': ('(a5*a6*b4*b1)/(a1*a4*b6)',),
        'a6': ('p17 * exp( p18 * V)',),
        'b6': ('p19 * exp(-p20 * V)',),
        'a7': ('p21 * exp(p22 * V)',),
        'b7': ('p23 * exp(-p24 * V)',),
    }

    auxiliary_expression = "p25 * {} * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_params_dict = {'E_Kr': -85}

    def __init__(self):
        super().__init__(states=self.states,
                         open_state=self.open_state,
                         rates=self.rates,
                         rate_dictionary=self.rate_dictionary,
                         auxiliary_expression=self.auxiliary_expression,
                         auxiliary_symbol=self.auxiliary_symbol,
                         shared_variables_dict=self.shared_variables_dict,
                         auxiliary_params_dict=self.auxiliary_params_dict)
