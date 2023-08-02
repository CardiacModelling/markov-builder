import numpy as np
import sympy as sp

from .MarkovChain import MarkovChain
from .rate_expressions import negative_rate_expr, positive_rate_expr


def construct_non_reversible_chain():
    """Construct a model structure that is known to not satisfy microscopic
    reversibiliy. This is used for testing.
    """
    mc = MarkovChain(name='non_reversible_example')

    mc.add_state('A')
    mc.add_state('D')
    mc.add_state('B', open_state=True)

    # Add rates
    rates = [('B', 'A', 'k1', 'k2'), ('A', 'D', 'k3', 'k4'), ('B', 'D', 'k5', 'k6')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_four_state_chain():
    """Construct and parameterise the model introduced by Beattie et al. in
    https://doi.org/10.1101/100677
    """

    mc = MarkovChain(name='Beattie_model')
    states = ['C', 'I', 'IC']

    for state in states:
        mc.add_state(state)

    mc.add_state('O', open_state=True)

    rates = [('O', 'C', 'k_2', 'k_1'), ('I', 'IC', 'k_2', 'k_1'),
             ('O', 'I', 'k_3', 'k_4'), ('C', 'IC', 'k_3', 'k_4')]

    for r in rates:
        mc.add_both_transitions(*r)

    # Model and parameters taken from https://doi.org/10.1113/JP275733
    rate_dictionary = {'k_1': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'k_2': negative_rate_expr + ((3.44E-5, 5.460E-2),),
                       'k_3': positive_rate_expr + ((0.0873, 8.91E-3),),
                       'k_4': negative_rate_expr + ((5.15E-3, 0.003158),)}

    mc.parameterise_rates(rate_dictionary)

    open_state = mc.get_state_symbol('O')
    auxiliary_expression = sp.sympify(f"g_Kr * {open_state} * (V - E_Kr)")

    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {'g_Kr': 0.1524,
                                    'E_Kr': -88})

    return mc


def construct_mazhari_chain():
    """Construct the Mazhari model structure for hERG as described in
    https://doi.org/10.1161/hh1301.093633
    """

    mc = MarkovChain(name='Mazhari_model')

    for state in ('C1', 'C2', 'C3', 'I'):
        mc.add_state(state)

    mc.add_state('O', open_state=True)

    rates = [('C1', 'C2', 'a0', 'b0'), ('C2', 'C3', 'kf', 'kb'), ('C3', 'O', 'a1', 'b1'),
             ('O', 'I', 'ai', 'bi'), ('I', 'C3', 'psi', 'ai3')]

    for r in rates:
        mc.add_both_transitions(*r)

    mc.substitute_rates({'psi': '(ai3*bi*b1)/(a1*ai)'})

    return mc


def construct_wang_chain():
    """Construct the Wang model structure for hERG as described in
    https://doi.org/10.1111/j.1469-7793.1997.045bl.x
    """
    mc = MarkovChain(name='Wang_model')

    mc.add_state('O', open_state=True)

    for state in ('C1', 'C2', 'C3', 'I'):
        mc.add_state(state)

    rates = [('C1', 'C2', 'a_a0', 'b_a0'), ('C2', 'C3', 'k_f', 'k_b'), ('C3', 'O', 'a_a1', 'b_a1'),
             ('O', 'I', 'a_1', 'b_1')]

    for r in rates:
        mc.add_both_transitions(*r)

    constant_rate_expr = ('a', ('a',))

    rate_dictionary = {'a_a0': positive_rate_expr + ((0.022348, 0.01176),),
                       'b_a0': negative_rate_expr + ((0.047002, 0.0631),),
                       'k_f': constant_rate_expr + ((0.023761,),),
                       'k_b': constant_rate_expr + ((0.036778,),),
                       'a_a1': positive_rate_expr + ((0.013733, 0.038198),),
                       'b_a1': negative_rate_expr + ((0.0000689, 0.04178),),

                       # Using 2mmol KCl values
                       'a_1': positive_rate_expr + ((0.090821, 0.023391),),
                       'b_1': negative_rate_expr + ((0.006497, 0.03268),)
                       }

    mc.parameterise_rates(rate_dictionary)

    open_state = mc.get_state_symbol('O')

    auxiliary_expression = sp.sympify(f"g_Kr * {open_state} * (V + E_Kr)")
    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {'g_Kr': 0.1524,
                                    'E_Kr': -88})
    return mc


def construct_HH_model(n: int, m: int, name: str = None):
    """ Construct a Markov model equivalent to a Hodgkin-Huxley conductance models

    :param n: The number of activation gates in the model
    :param m: The number of inactivation gates in the model

    :return: A MarkovChain with n x m states

    """

    if n < 2 or m < 2:
        raise Exception()

    if name is None:
        name = f"HH_{n}_{m}"

    labels = []
    for i in range(n):
        for j in range(m):
            if i == 0:
                if j == 0:
                    label = 'O'
                else:
                    label = f"C{j}"
            elif j == 0:
                label = f"I{i}"
            else:
                label = f"I{i}C{j}"
            labels.append(label)

    mc = MarkovChain(name=name)

    for label in labels:
        if label == 'O':
            mc.add_state(label, open_state=True)
        else:
            mc.add_state(label)

    labels = np.array(labels, dtype=object).reshape((n, m))

    # Add inactivation transitions
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                mc.add_both_transitions(labels[i, j], labels[i + 1, j], sp.sympify(f"{n-i-1} * b_o"),
                                        sp.sympify(f"{i+1}*a_o"))
            if j < m - 1:
                mc.add_both_transitions(labels[i, j], labels[i, j + 1], sp.sympify(f"{m-j-1} * b_i"),
                                        sp.sympify(f"{j+1}*a_i"))
    return mc


def construct_kemp_model():
    """Construct and parameterise the model introduced by Kemp et al. in
    https://doi.org/10.1085/jgp.202112923
    """

    mc = MarkovChain(name='Kemp_model')

    # Now the conducting state
    mc.add_state('O', open_state=True)

    # First add the non-conducting states
    for state in ('IO', 'C1', 'IC1', 'C2', 'IC2'):
        mc.add_state(state)

    rates = [
        ('O', 'IO', 'b_h', 'a_h'), ('C1', 'IC1', 'b_h', 'a_h'), ('C2', 'IC2', 'b_h', 'a_h'),
        ('O', 'C1', 'b_2', 'a_2'), ('C1', 'C2', 'b_1', 'a_1'),
        ('IO', 'IC1', 'b_2', 'a_2'), ('IC1', 'IC2', 'b_1', 'a_1')
    ]

    for r in rates:
        mc.add_both_transitions(*r)

    rate_dictionary = {
        # Activation rates
        'a_1': positive_rate_expr + ((8.53e-03, 8.32e-02),),
        'a_2': positive_rate_expr + ((1.49e-01, 2.43e-02),),

        # Deactivation rates
        'b_1': negative_rate_expr + ((1.26e-02, 1.04e-04),),
        'b_2': negative_rate_expr + ((5.58e-04, 4.07e-02),),

        # Recovery rate
        'a_h': negative_rate_expr + ((7.67e-02, 2.25e-02),),

        # Inactivation rate
        'b_h': positive_rate_expr + ((2.70e-01, 1.58e-02),),
    }

    open_state = mc.get_state_symbol('O')

    auxiliary_expression = sp.sympify(f"g_Kr * {open_state} * (V + E_Kr)")
    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {
                                       'g_Kr': 7.05e-02,  # Use conductance from Cell 2
                                       'E_Kr': -88,  # -88mV chosen arbitrarily
                                   })
    mc.parameterise_rates(rate_dictionary)
    return mc
