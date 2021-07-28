#!/usr/bin/env python
#
# Quick experiment with building markov models
#
#
#
from __future__ import print_function
import myokit

class State(object):
    def __init__(self, name, indice, conducting=False):
        self.name = str(name)
        self.conducting = bool(conducting)
        self.indice = None


class Rate(object):
    def __init__(self, name, index, positive, alpha=1e-4, beta=1e-2):
        self.name = str(name)
        self.positive = bool(positive)
        assert(alpha > 0)
        assert(beta > 0)
        self.alpha = float(alpha)
        self.beta = float(beta)


class Edge(object):
    def __init__(self, state_from, state_to,
                 forward_rate, forward_multiplier,
                 backward_rate, backward_multiplier):
        self.state_from = state_from
        self.state_to = state_to
        self.forward_rate = forward_rate
        self.forward_multiplier = forward_multiplier
        self.backward_rate = backward_rate
        self.backward_multiplier = backward_multiplier


class MarkovModel(object):
    def __init__(self):

        self.states = []
        self.state_names = {}   # Check for unique state names
        self.rates = []
        self.rate_names = {}    # Check for unique rate names
        self.edges = []         #

    def add_state(self, name, conducting=False):
        if name in self.state_names:
            raise ValueError('Duplicate state name "' + name + '".')
        state = State(name, conducting)
        state.indice = len(self.states)
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

    def dfs_spanning_tree(self, root=None):
        """
        Finds a spanning tree by performing a depth first search using a recursive approach
        """
        if root is None:
            root = self.states[0]

        graph = [self.states, self.edges]

        # TODO move this elsewhere
        def remove_node(graph, n):
            # remove nodes from graph
            if n in graph[0]:
                graph[0] = list(filter(lambda node : n != node, graph[0]))
            for e in graph[1]:
                graph[1] = list(filter(lambda e: e.state_from != n and e.state_to != n, graph[1]))
            return graph


        def dfs_branch(graph, node):
            edges = list(filter(lambda e: e.state_from==node or e.state_to==node, graph[1]))
            tree=[[node], edges]
            print(edges)
            unexplored_graph = remove_node(graph, node)
            for edge in edges:
                print("here")
                print(tree)
                new_node = edge.state_to if edge.state_to != node else edge.state_from
                print(new_node.name)
                if new_node not in unexplored_graph[0]:
                    continue
                branch = dfs_branch(unexplored_graph, new_node)
                print("branch is: ")
                for e in branch[1]:
                    print("from {} to {}".format(e.state_from.name, e.state_to.name))
                tree[0] = tree[0] + branch[0]
                tree[1] = tree[1] + branch[1]

                for node in tree:
                    unexplored_graph = remove_node(unexplored_graph, node)
            return tree
        return dfs_branch(graph, root)

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
            Edge(state1, state2, forward, fm, backward, bm))

    def show(self):
        print('='*40)
        print('States: ' + str(len(self.states)))
        print('Rates: ' + str(len(self.rates)))
        print('Edges: ' + str(len(self.edges)))
        print('-'*40)
        for edge in self.edges:
            fwn = str(edge.forward_multiplier) + edge.forward_rate.name
            bwn = str(edge.backward_multiplier) + edge.backward_rate.name
            fwn += ' ' * (max(len(fwn), len(bwn)) - len(fwn))
            bwn += ' ' * (max(len(fwn), len(bwn)) - len(bwn))
            print(
                edge.state_from.name + ' > '
                + fwn + ' > ' + edge.state_to.name)
            print(
                ' ' * len(edge.state_from.name) + ' < '
                + bwn + ' < ')

    def n_parameters(self):
        return len(self._edges) * 2

    def model(self, E=50, g=1, component='ikr', current='IKr'):
        m = myokit.Model()

        # Add time variable
        ce = m.add_component('engine')
        t = ce.add_variable('time')
        t.set_rhs(0)
        t.set_binding('time')

        # Add membrane potential variable
        cm = m.add_component('membrane')
        v = cm.add_variable('V')
        v.set_rhs(-80)
        v.set_label('membrane_potential')

        # Add current component
        cc = m.add_component(component)
        cc.add_alias('V', v)

        # Add parameters
        i = 0
        pvars = []
        for rate in self.rates:
            i += 1
            p = cc.add_variable('p' + str(i))
            p.set_rhs(rate.alpha)
            pvars.append(p)
            i += 1
            p = cc.add_variable('p' + str(i))
            p.set_rhs(rate.beta)
            pvars.append(p)
        i += 1
        p = cc.add_variable('p' + str(i))
        p.set_rhs(myokit.Number(g))
        pvars.append(p)

        # Add rates, store in map from names to variables
        rvars = {}
        for i, rate in enumerate(self.rates):
            var = cc.add_variable(rate.name)
            rvars[rate.name] = var

            # Generate term a * exp(+-b * V)
            alpha = myokit.Name(pvars[2 * i])
            beta = myokit.Name(pvars[1 + 2 * i])
            if not rate.positive:
                beta = myokit.PrefixMinus(beta)
            var.set_rhs(myokit.Multiply(
                alpha, myokit.Exp(myokit.Multiply(beta, myokit.Name(v)))))

        # Add reversal potential variable
        e = cc.add_variable('E')
        e.set_rhs(myokit.Number(E))

        # Add maximum conductance variable
        g = cc.add_variable('g')
        g.set_rhs(myokit.Name(p))

        # Create states, store in map from names to variables
        svars = {}
        for i, state in enumerate(self.states):
            var = cc.add_variable(state.name)
            var.promote(0)
            svars[state.name] = var

        # Set equations for states
        for state in self.states:

            incoming = []
            outgoing = []
            for edge in self.edges:
                # Gather info
                if state == edge.state_from:
                    # Edge from this state to other state
                    state2 = edge.state_to
                    rate_in = edge.backward_rate
                    rate_out = edge.forward_rate
                    mult_in = edge.backward_multiplier
                    mult_out = edge.forward_multiplier
                elif state == edge.state_to:
                    # Edge from other state to this state
                    state2 = edge.state_from
                    rate_in = edge.forward_rate
                    rate_out = edge.backward_rate
                    mult_in = edge.forward_multiplier
                    mult_out = edge.backward_multiplier
                else:
                    continue

                # Add incoming term
                term = myokit.Name(svars[state2.name])
                term = myokit.Multiply(
                    myokit.Name(rvars[rate_in.name]), term)
                if mult_in != 1:
                    term = myokit.Multiply(myokit.Number(mult_in), term)
                incoming.append(term)

                # Add outgoing term
                term = myokit.Name(rvars[rate_out.name])
                if mult_out != 1:
                    term = myokit.Multiply(myokit.Number(mult_out), term)
                outgoing.append(term)

            # Start with outgoing terms (grouped)
            outgoing = iter(outgoing)
            rhs = next(outgoing)
            for term in outgoing:
                rhs = myokit.Plus(rhs, term)
            rhs = myokit.PrefixMinus(rhs)
            rhs = myokit.Multiply(rhs, myokit.Name(svars[state.name]))

            # Add incoming terms (one by one)
            incoming = iter(incoming)
            for term in incoming:
                rhs = myokit.Plus(rhs, term)

            # Set rhs
            svars[state.name].set_rhs(rhs)

        # Add current variable
        var = cc.add_variable(current)
        rhs = myokit.Name(g)
        for state in self.states:
            if state.conducting:
                rhs = myokit.Multiply(rhs, myokit.Name(state_map[state.name]))
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
m.connect(c1, ii, a, b, 1, 3)

spanning_tree = m.dfs_spanning_tree(c1)
print(len(spanning_tree))

m.show()


print(spanning_tree)
print(spanning_tree[0][-1].name)
for e in spanning_tree[1]:
    print("from {} to {}".format(e.state_from.name, e.state_to.name))


if len(spanning_tree[0]) == len(m.states):
    print("Graph is connected. Minimum spanning tree consists of {} edges and has {} nodes".format(len(spanning_tree[1]), len(spanning_tree[0])))
    print("there are {} states".format(len(m.states)))
else:
    print("Spanning tree has {} edges and contains {} nodes".format(len(spanning_tree[1]), len(spanning_tree[0])))
    assert(False)
