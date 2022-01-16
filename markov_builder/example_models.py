import sympy as sp
import numpy as np

from .MarkovChain import MarkovChain


def construct_M10_chain():
    mc = MarkovChain(name='M10')

    mc.add_states(('IC1', 'IC2', 'IO', 'C1', 'C2'))
    mc.add_state('s_O', open_state=True)
    rates = [('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'),
             ('IO', 's_O', 'ah', 'bh'), ('s_O', 'C1', 'b2', 'a2'),
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
    states = [('s_O', {'open_state': True}), ('C'), ('s_I'), ('IC')]
    mc.add_states(states)

    rates = [('s_O', 'C', 'k_2', 'k_1'), ('s_I', 'IC', 'k_1', 'k_2'), ('IC', 's_I', 'k_1', 'k_2'),
             ('s_O', 's_I', 'k_3', 'k_4'), ('C', 'IC', 'k_3', 'k_4')]

    for r in rates:
        mc.add_both_transitions(*r)

    # Model and parameters taken from https://doi.org/10.1113/JP275733
    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))

    rate_dictionary = {'k_1': positive_rate_expr + ((2.26E-4, 6.99E-2),),
                       'k_2': negative_rate_expr + ((3.44E-5, 5.460E-2),),
                       'k_3': positive_rate_expr + ((0.0873, 8.91E-3),),
                       'k_4': negative_rate_expr + ((5.15E-3, 0.003158),)}

    mc.parameterise_rates(rate_dictionary, shared_variables=('V',))

    auxiliary_expression = sp.sympify('g_Kr * s_O * (V + E_Kr)')

    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {'g_Kr': 0.1524,
                                    'E_Kr': -88})

    return mc


def construct_mazhari_chain():
    mc = MarkovChain(name='Mazhari_model')

    mc.add_states(('C1', 'C2', 'C3', 's_I'))
    mc.add_state('s_O', open_state=True)

    rates = [('C1', 'C2', 'a0', 'b0'), ('C2', 'C3', 'kf', 'kb'), ('C3', 's_O', 'a1', 'b1'),
             ('s_O', 's_I', 'ai', 'bi'), ('s_I', 'C3', 'psi', 'ai3')]

    for r in rates:
        mc.add_both_transitions(*r)

    mc.substitute_rates({'psi': '(ai3*bi*b1)/(a1*ai)'})

    return mc


def construct_wang_chain():
    mc = MarkovChain(name='Wang_model')

    mc.add_states(('C1', 'C2', 'C3', 's_I'))
    mc.add_state('s_O', open_state=True)

    rates = [('C1', 'C2', 'a', 'b'), ('C2', 'C3', 'c', 'd'), ('C3', 's_O', 'e', 'f'),
             ('s_O', 's_I', 'g', 'h')]

    for r in rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    constant_rate_expr = ('a', ('a',))

    rate_dictionary = {'a': positive_rate_expr,
                       'b': negative_rate_expr,
                       'c': constant_rate_expr,
                       'd': constant_rate_expr,
                       'e': positive_rate_expr,
                       'f': negative_rate_expr,
                       'g': positive_rate_expr,
                       'h': negative_rate_expr
                       }

    mc.parameterise_rates(rate_dictionary, shared_variables=('V',))

    auxiliary_expression = sp.sympify('g_Kr * s_O * (V + E_Kr)')
    mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                   {'g_Kr': 0.1524,
                                    'E_Kr': -88})

    for r in rates:
        mc.add_both_transitions(*r)

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
                    label = 's_O'
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
