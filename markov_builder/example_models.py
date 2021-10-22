from .MarkovChain import MarkovChain


def construct_M10_chain():
    mc = MarkovChain(name='M10')

    mc.add_states(('IC1', 'IC2', 'IO', 'C1', 'C2'))
    mc.add_state('s_O', open=True)
    rates = [('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'), ('IO', 's_O', 'ah', 'bh'), ('s_O', 'C1',
             'b2', 'a2'), ('C1', 'C2', 'b1', 'a1'), ('C2', 'IC2', 'bh', 'ah'), ('C1', 'IC1', 'bh', 'ah')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_non_reversible_chain():
    mc = MarkovChain(name='non_reversible_example')

    mc.add_states(('A', 'D'))
    mc.add_state('B', open=True)
    rates = [('B', 'A', 'k1', 'k2'), ('A', 'D', 'k3', 'k4'), ('B', 'D', 'k5', 'k6')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_four_state_chain():
    mc = MarkovChain(name='Beattie_model')
    rates = ['k{}'.format(i) for i in [1, 2, 3, 4]]
    mc.add_rates(rates)
    states = [('s_O', True), ('C', False), ('s_I', False), ('IC', False)]
    mc.add_states(states)
    mc.add_state('s_O', open=True)

    rates = [('s_O', 'C', 'k2', 'k1'), ('s_I', 'IC', 'k1', 'k2'), ('IC', 's_I', 'k1', 'k2'),
             ('s_O', 's_I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc


def construct_mazhari_chain():
    mc = MarkovChain(name='Mazhari_model')

    mc.add_states(('C1', 'C2', 'C3', 's_I'))
    mc.add_state('s_O', open=True)

    rates = [('C1', 'C2', 'a0', 'b0'), ('C2', 'C3', 'kf', 'kb'), ('C3', 's_O', 'a1', 'b1'),
             ('s_O', 's_I', 'ai', 'bi'), ('s_I', 'C3', 'psi', 'ai3')]

    for r in rates:
        mc.add_both_transitions(*r)

    mc.substitute_rates({'psi': '(ai3*bi*b1)/(a1*ai)'})

    return mc


def construct_wang_chain():
    mc = MarkovChain(name='Wang_model')

    mc.add_states(('C0', 'C1', 'C2', 's_I'))
    mc.add_state('s_O', open=True)

    rates = [('C0', 'C1', 'aa0', 'ba0'), ('C1', 'C2', 'kf', 'kb'), ('C2', 's_O', 'aa1', 'ba1'),
             ('s_O', 's_I', 'ai', 'bi')]

    for r in rates:
        mc.add_both_transitions(*r)

    return mc
