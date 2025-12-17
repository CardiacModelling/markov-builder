from numpy import nan
from markov_builder.MarkovChain import MarkovChain


class model_23(MarkovChain):
    description = "IKr Markov model 23 (Sanmitra Ghosh)"

    states = (
        'C3', 'C2', 'C1', 'O',
        'I', 'IC1', 'IC2', 'IC3'
    )

    open_state = 'O'

    rates = [
        ('C3', 'C2', 'a1', 'b1'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C1', 'O',  'a3', 'b3'),
        ('O',  'I',  'a4', 'b4'),

        ('C3', 'IC3', 'a7', 'b7'),
        ('C2', 'IC2', 'a6', 'b6'),
        ('C1', 'IC1', 'a5', 'b5'),

        ('IC3', 'IC2', 'a8', 'b8'),
        ('IC2', 'IC1', 'a9', 'b9'),
        ('IC1', 'I',   'a10', 'b10'),
    ]

    rate_dictionary = {
        'a1': ('p1 * exp(p2 * V)',),
        'b1': ('p3 * exp(-p4 * V)',),
        'a2': ('p5 * exp(p6 * V)',),
        'b2': ('p7 * exp(-p8 * V)',),
        'a3': ('p9 * exp(p10 * V)',),
        'b3': ('p11 * exp(-p12 * V)',),
        'a4': ('p13 * exp(p14 * V)',),
        'b4': ('p15 * exp(-p16 * V)',),

        'a8': ('p17 * exp(p18 * V)',),
        'b8': ('p19 * exp(-p20 * V)',),
        'a9': ('p21 * exp(p22 * V)',),
        'b9': ('p23 * exp(-p24 * V)',),
        'a10': ('p25 * exp(p26 * V)',),
        'b10': ('p27 * exp(-p28 * V)',),

        'a5': ('p29 * exp(p30 * V)',),
        'b5': ('(a10 * a5 * b4 * b3) / (a3 * a4 * b10)',),
        'a6': ('p31 * exp(p32 * V)',),
        'b6': ('(a9 * a6 * b5 * b2) / (a2 * a5 * b9)',),
        'a7': ('p33 * exp(p34 * V)',),
        'b7': ('(a8 * a7 * b1 * b6) / (a1 * a6 * b8)',),
    }

    shared_variables_dict = {
        'V': nan,

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
        'p13': 0.08730,
        'p14': 8.91e-3,
        'p15': 5.15e-3,
        'p16': 0.03158,
        'p17': 0.08730,
        'p18': 8.91e-3,
        'p19': 5.15e-3,
        'p20': 0.03158,
        'p21': 0.06990,
        'p22': 3.45e-5,
        'p23': 0.05462,
        'p24': 0.08730,
        'p25': 8.91e-3,
        'p26': 5.15e-3,
        'p27': 0.03158,
        'p28': 0.08730,
        'p29': 8.91e-3,
        'p30': 5.15e-3,
        'p31': 0.03158,
        'p32': 5.15e-3,
        'p33': 0.03158,
        'p34': 0.03158,
        'p35': 0.15240,
    }

    auxiliary_expression = 'p35 * state_O * (V - E_Kr)'
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {
        'E_Kr': -85,
    }

    def __init__(self):
        super().__init__(
            states=self.states,
            transition_rates=self.rates,
            rate_expressions=self.rate_dictionary,
            auxiliary_expression=self.auxiliary_expression,
            auxiliary_symbol=self.auxiliary_symbol,
            shared_variables=self.shared_variables_dict,
            auxiliary_parameters=self.auxiliary_parameters,
        )

        self.set_state_attribute('O', open_state=True)
