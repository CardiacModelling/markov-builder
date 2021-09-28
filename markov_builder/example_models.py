from . MarkovChain import MarkovChain


def construct_M10_chain():
    mc = MarkovChain()

    mc.add_states(('IC1', 'IC2', 'IO', 'C1', 'C2'))
    mc.add_state('O', open=True)
    rates = (('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'), ('IO', 'O', 'ah', 'bh'), ('O', 'C1',
             'b2', 'a2'), ('C1', 'C2', 'b1', 'a1'), ('C2', 'IC2', 'bh', 'ah'), ('C1', 'IC1', 'bh', 'ah'))

    for r in rates:
        mc.add_both_transitions(*r)
    return mc


def construct_non_reversible_chain():
    mc = MarkovChain()

    mc.add_states(('A', 'D'))
    mc.add_state('B', open=True)
    rates = (('B', 'A', 'k1', 'k2'), ('A', 'D', 'k3', 'k4'), ('B', 'D', 'k5', 'k6'))

    for r in rates:
        mc.add_both_transitions(*r)
    return mc


def construct_four_state_chain():
    mc = MarkovChain()
    rates = ['k{}'.format(i) for i in [1, 2, 3, 4]]
    mc.add_rates(rates)
    states = [('O', True), ('C', False), ('I', False), ('IC', False)]
    mc.add_states(states)
    mc.add_state('O', open=True)

    rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'),
             ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

    for r in rates:
        mc.add_both_transitions(*r)
    return mc
