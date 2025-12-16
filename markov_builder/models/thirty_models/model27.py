from numpy import nan
from markov_builder.MarkovChain import MarkovChain

class model_27(MarkovChain):
    description = "IKr Markov model 27 (Sanmitra Ghosh) with Om and h gates"

    states = ('C1', 'C2', 'C3', 'C4', 'Om', 'Oh', 'I')
    rates = [
        ('C1', 'Om', 'a1', 'b1'),
        ('C2', 'C1', 'a2', 'b2'),
        ('C3', 'C2', 'a3', 'b3'),
        ('C3', 'C4', 'a4', 'b4'),
        ('I', 'Oh', 'ah', 'bh')
    ]

    rate_dictionary = {
        'a1': ('p1 * exp(p2*V)',),
        'b1': ('p3 * exp(-p4*V)',),
        'ah': ('p7 * exp(-p8*V)',),
        'bh': ('p5 * exp(p6*V)',),
        'a2': ('p9 * exp(p10*V)',),
        'b2': ('p11 * exp(-p12*V)',),
        'a3': ('p13 * exp(p14*V)',),
        'b3': ('p15 * exp(-p16*V)',),
        'a4': ('p17 * exp(p18*V)',),
        'b4': ('p19 * exp(-p20*V)',),
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
        'p9': 0.08730,
        'p10': 8.91e-3,
        'p11': 5.15e-3,
        'p12': 0.03158,
        'p13': 0.08730,
        'p14': 8.91e-3,
        'p15': 5.15e-3,
        'p16': 0.03158,
        'p17': 5.15e-3,
        'p18': 0.03158,
        'p19': 5.15e-3,
        'p20': 0.03158,
        'p21': 0.15240,
    }

    open_state = 'Om'  # the Markov open state representing the m gate
    auxiliary_expression = 'p21 * state_Om * state_Oh * (V - E_Kr)'
    auxiliary_symbol = 'I_Kr'

    auxiliary_parameters = {'E_Kr': -85}

    def __init__(self):
        super().__init__(
            states=self.states,
            open_state=self.open_state,
            transition_rates=self.rates,
            rate_expressions=self.rate_dictionary,
            auxiliary_expression=self.auxiliary_expression,
            auxiliary_symbol=self.auxiliary_symbol,
            shared_variables=self.shared_variables_dict,
            auxiliary_parameters=self.auxiliary_parameters
        )
