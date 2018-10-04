#!/usr/bin/env python
#
# Quick experiment with building markov models
#
#
#
from __future__ import print_function
import myokit

class State(object):
    def __init__(self, name, conducting=False):
        self.name = str(name)
        self.conducting = bool(conducting)

class Rate(object):
    def __init__(self, name, index, positive):
        self.name = str(name)
        self.positive = bool(positive)

class MarkovModel(object):
    def __init__(self):

        self.states = []
        self.state_names = {}
        self.rates = []
        self.rate_names = {}
        self.edges = []

    def add_state(self, name, conducting=False):
        if name in self.state_names:
            raise ValueError('Duplicate state name "' + name + '".')
        state = State(name, conducting)
        self.state_names[name] = state
        self.states.append(state)
        return state

    def add_rates(self, positive, negative):
        """
        Adds two rates to the model, one with a positive V relation, one with a
        negative V relation.
        """
        if positive in self.rate_names:
            raise ValueError('Duplicate parameter name "' + positive + '".')
        if negative in self.rate_names:
            raise ValueError('Duplicate parameter name "' + negative + '".')
        n = len(self.rates)
        r1 = Rate(positive, n, True)
        r2 = Rate(negative, n + 1, False)
        self.rates.append(r1)
        self.rates.append(r2)
        self.rate_names[positive] = r1
        self.rate_names[negative] = r2
        return r1, r2

    def connect(self, state1, state2, forward, backward, fm=1, bm=1):
        """
        Connects two states. The transition from state1 to state2 should be
        given as ``forward``, and the reverse as ``backward``. If ``forward``
        and ``backward`` are ``Rate`` objects, these will be used. If they are
        strings, new rates will be created, such that ``forward`` has a
        positive V relation.

        Multipliers for the rates can be given with ``fm`` and ``bm``.
        """
        str1 = isinstance(forward, str)
        str2 = isinstance(backward, str)
        if str1 and str2:
            forward, backward = self.add_rates(forward, backward)
        elif str1 or str2:
            raise ValueError(
                'Rates must both be strings, or both be Rate objects.')
        elif not (isinstance(forward, Rate) and isinstance(backward, Rate)):
            raise ValueError('Rates can only be strings or Rate objects.')
        self.edges.append(
            (state1, state2, forward, backward, fm, bm))

    def show(self):
        print('='*40)
        print('States: ' + str(len(self.states)))
        print('Rates: ' + str(len(self.rates)))
        print('Edges: ' + str(len(self.edges)))
        print('-'*40)
        for edge in self.edges:
            s1, s2, fw, bw, fm, bm = edge
            fwn = str(fm) + fw.name
            bwn = str(bm) + bw.name
            fwn += ' ' * (max(len(fwn), len(bwn)) - len(fwn))
            bwn += ' ' * (max(len(fwn), len(bwn)) - len(bwn))
            print(s1.name + ' > ' + fwn + ' > ' + s2.name)
            print(' ' * len(s1.name) + ' < ' + bwn + ' > ')

    def n_parameters(self):
        return len(self._edges) * 2

    def model(self, E=50, g=1, comp='ikr', var='IKr'):
        m = myokit.Model()

        ce = m.add_component('engine')
        t = ce.add_variable('time')
        t.set_rhs(0)
        t.set_binding('time')

        cm = m.add_component('membrane')
        v = cm.add_variable('V')
        v.set_rhs(-80)
        v.set_label('membrane_potential')

        cc = m.add_component(comp)
        e = cc.add_variable('E')
        e.set_rhs(E)
        g = cc.add_variable('g')
        g.set_rhs(g)

        iconducting = []
        states = []
        for i, x in enumerate(self.states):
            states.append(cc.add_variable(x.name))
            if x.conducting:
                iconducting.append(i)

        var = cc.add_variable(var)
        rhs = myokit.Name(g)
        conducting = [x for x in self.states if x.conducting]
        for i in iconducting:
            rhs = myokit.Multiply(rhs, myokit.Name(states[i]))
        rhs = myokit.Multiply(
            rhs, myokit.Minus(myokit.Name(v), myokit.Name(e)))
        var.set_rhs(rhs)

        print(m.code())

    def rate_dict(self, parameters):
        """
        Generate a dictionary connecting rates to parameters.
        """
        assert(len(parameters) == 2 * len(self.rates) + 1)

        param = iter(parameters)
        rate_dict = {}
        for rate in self.rates:
            rate_dict[rate.name] = [next(param), next(param), rate.positive]
        return rate_dict


# Test

#     3a     2a     1a    c
# C3 --- C2 --- C1 --- O --- I
#     b      2b     3b    d
#
m = MarkovModel()
c3 = m.add_state('C3')
c2 = m.add_state('C2')
c1 = m.add_state('C1')
oo = m.add_state('O', True)
ii = m.add_state('I')
a, b = m.add_rates('a', 'b')
m.connect(c3, c2, a, b, 3, 1)
m.connect(c2, c1, a, b, 2, 2)
m.connect(c1, oo, a, b, 1, 3)
m.connect(oo, ii, 'c', 'd')


m.show()

m.model()
