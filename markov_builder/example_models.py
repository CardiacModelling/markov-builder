import numpy as np
import sympy as sp

from .MarkovChain import MarkovChain
from .rate_expressions import negative_rate_expr, positive_rate_expr


def construct_M10_chain():
    mc = MarkovChain(name='M10')

    mc.add_states(('IC1', 'IC2', 'IO', 'C1', 'C2'))
    mc.add_state('O', open_state=True)
    rates = [('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'),
             ('IO', 'O', 'ah', 'bh'), ('O', 'C1', 'b2', 'a2'),
             ('C1', 'C2', 'b1', 'a1'), ('C2', 'IC2', 'bh', 'ah'),
             ('C1', 'IC1', 'bh', 'ah')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_non_reversible_chain():
    mc = MarkovChain(name='non_reversible_example')

    mc.add_states(('A', 'D'))
    mc.add_state('B', open_state=True)
    rates = [('B', 'A', 'k1', 'k2'), ('A', 'D', 'k3', 'k4'), ('B', 'D', 'k5', 'k6')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_four_state_chain():
    mc = MarkovChain(name='Beattie_model')
    states = [('O', {'open_state': True}), ('C'), ('I'), ('IC')]
    mc.add_states(states)

    rates = [('O', 'C', 'k_2', 'k_1'), ('I', 'IC', 'k_2', 'k_1'),
             ('O', 'I', 'k_3', 'k_4'), ('C', 'IC', 'k_3', 'k_4')]

    for r in rates:
        mc.add_both_transitions(*r)

    # Model and parameters taken from https://doi.org/10.1113/JP275733
    rate_dictionary = {'k_1': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'k_2': negative_rate_expr + ((3.44E-5, 5.460E-2),),
                       'k_3': positive_rate_expr + ((0.0873, 8.91E-3),),
                       'k_4': negative_rate_expr + ((5.15E-3, 0.003158),)}

    mc.parameterise_rates(rate_dictionary, shared_variables=('V',))

    open_state = mc.get_state_symbol('O')
    auxiliary_expression = sp.sympify(f"g_Kr * {open_state} * (V - E_Kr)")

    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {'g_Kr': 0.1524,
                                    'E_Kr': -88})

    return mc


def construct_mazhari_chain():
    mc = MarkovChain(name='Mazhari_model')

    mc.add_states(('C1', 'C2', 'C3', 'I'))
    mc.add_state('O', open_state=True)

    rates = [('C1', 'C2', 'a0', 'b0'), ('C2', 'C3', 'kf', 'kb'), ('C3', 'O', 'a1', 'b1'),
             ('O', 'I', 'ai', 'bi'), ('I', 'C3', 'psi', 'ai3')]

    for r in rates:
        mc.add_both_transitions(*r)

    mc.substitute_rates({'psi': '(ai3*bi*b1)/(a1*ai)'})

    return mc


def construct_wang_chain():
    mc = MarkovChain(name='Wang_model')

    mc.add_states(('C1', 'C2', 'C3', 'I'))
    mc.add_state('O', open_state=True)

    rates = [('C1', 'C2', 'a_a0', 'b_a0'), ('C2', 'C3', 'k_f', 'k_b'), ('C3', 'O', 'a_a1', 'b_a1'),
             ('O', 'I', 'a_1', 'b_1')]

    for r in rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    constant_rate_expr = ('a', ('a',))

    rate_dictionary = {'a_a0': positive_rate_expr,
                       'b_a0': negative_rate_expr,
                       'k_f': constant_rate_expr,
                       'k_b': constant_rate_expr,
                       'a_a1': positive_rate_expr,
                       'b_a1': negative_rate_expr,
                       'a_1': positive_rate_expr,
                       'b_1': negative_rate_expr
                       }

    mc.parameterise_rates(rate_dictionary, shared_variables=('V',))

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
        mc.add_state(label)

    labels = np.array(labels, dtype=object).reshape((n, m))

    # Add inactivation transitions
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                mc.add_both_transitions(labels[i, j], labels[i+1, j], sp.sympify(f"{n-i-1} * b_o"),
                                        sp.sympify(f"{i+1}*a_o"))
            if j < m - 1:
                mc.add_both_transitions(labels[i, j], labels[i, j+1], sp.sympify(f"{m-j-1} * b_i"),
                                        sp.sympify(f"{j+1}*a_i"))
    return mc
