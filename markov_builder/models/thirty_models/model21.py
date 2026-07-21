from numpy import nan

from markov_builder.MarkovChain import MarkovChain


class model_21(MarkovChain):
    description = ""
    states = ('O', 'C1', 'C2', 'C3', 'I', 'IC1', 'IC2', 'IC3')
    rates = [
        ('C1', 'O', 'a3', 'b3'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C3', 'C2', 'a1', 'b1'),

        ('IC1', 'I', 'a6', 'b6'),
        ('IC2', 'IC1', 'a5', 'b5'),
        ('IC3', 'IC2', 'a4', 'b4'),

        ('C3', 'IC3', 'ah', 'bh'),
        ('C2', 'IC2', 'ah', 'bh'),
        ('C1', 'IC1', 'ah', 'bh'),
        ('O', 'I', 'ah', 'bh'),
    ]

    shared_variables_dict = {'V': nan,
                             'p1': 2.26e-4,
                             'p2': 0.06990,
                             'p3': 3.45e-5,
                             'p4': 0.05462,
                             'p5': 0.08730,
                             'p6': 8.91e-3,
                             'p7': 5.15e-3,
                             'p8': 0.03158,
                             'p9': 2.26e-4,
                             'p10': 0.0699,
                             'p11': 3.45e-5,
                             'p12': 0.05462,
                             'p13': 0.0873,
                             'p14': 8.91e-3,
                             'p15': 5.15e-3,
                             'p16': 0.03158,
                             'p17': 0.08730,
                             'p18': 8.91e-3,
                             'p19': 5.15e-3,
                             'p20': 0.03158,
                             'p21': 5.15e-3,
                             'p22': 0.03158,
                             'p23': 0.15240,
                             }

    rate_dictionary = {
        'ah': ('p1 * exp(p2*V)',),
        'bh': ('p3 * exp(-p4*V)',),
        'a1': ('p5 * exp(p6*V)',),
        'b1': ('p7 * exp(-p8*V)',),
        'a2': ('p9 * exp(p10*V)',),
        'b2': ('p11 * exp(-p12*V)',),
        'a3': ('p13 * exp(p14*V)',),
        'b3': ('p15 * exp(-p16*V)',),
        'a4': ('p17 * exp(p18*V)',),
        'b4': ('(a4*b1/a1)',),
        'a5': ('p19 * exp(p20*V)',),
        'b5': ('(a5*b2/a2)',),
        'a6': ('p21 * exp(p22*V)',),
        'b6': ('(a6*b3/a3)',)
    }

    auxiliary_expression = "p23 * state_O * (V - E_Kr)"
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {'E_Kr': -85}

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
