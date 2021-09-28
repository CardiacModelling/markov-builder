import itertools
import logging
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp
from numpy.random import default_rng


class MarkovChain():
    def __init__(self, states: Union[list, None] = None, seed: Union[int, None] = None):
        # Needs to be a MultiDiGraph to visualise properly (??)
        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.rates = []
        self.rng = default_rng(seed)

    def mirror_model(self, prefix: str, new_rates: bool = False):
        trapped_graph = nx.relabel_nodes(self.graph, dict([(n, "{}{}".format(prefix, n)) for n in self.graph.nodes]))
        nx.set_node_attributes(trapped_graph, False, 'open')

        if new_rates:
            for frm, to, attr in self.graph.edges(data=True):
                new_rate = prefix + attr['rate']
                if new_rate not in self.rates:
                    self.rates.append(new_rate)
                attr['rate'] = new_rate

        new_graph = nx.compose(trapped_graph, self.graph)
        # Get open state name
        open_nodes = [n for n, d in new_graph.nodes(data=True) if d['open']]
        assert(len(open_nodes) == 1)

        self.graph = new_graph

    def add_open_trapping(self, prefix: str = "d_", new_rates: bool = False):
        self.mirror_model(prefix, new_rates)
        self.add_rates(("drug_on", "drug_off"))
        open_nodes = [n for n, d in self.graph.nodes(data=True) if d['open']]
        self.add_both_transitions(open_nodes[0], "d_{}".format(open_nodes[0]), 'drug_on', 'drug_off')

    def add_state(self, label: Union[str, None], open: bool = False):
        if label is None:
            # TODO Generate new label name
            label = 'undefined'
        self.graph.add_node(label, open=open)

    def add_states(self, states: list):
        for state in states:
            if isinstance(state, str):
                self.add_state(state)
            else:
                self.add_state(*state)

    def add_rate(self, rate: str):
        if rate in self.rates:
            # TODO
            raise Exception()
        else:
            self.rates = self.rates + [rate]

    def add_rates(self, rates: list):
        for rate in rates:
            self.add_rate(rate)

    def add_both_transitions(self, frm: str, to: str, fwd_rate: Union[str, sp.Expr, None], bwd_rate: Union[str, None]):
        self.add_transition(frm, to, fwd_rate)
        self.add_transition(to, frm, bwd_rate)

    def add_transition(self, from_node: str, to_node: str, transition_rate: Union[str, None]):
        # TODO: Nice exceptions
        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            raise Exception("A node wasn't present in the graph ({} or {})".format(from_node, to_node))

        if not isinstance(transition_rate, str):
            transition_rate = str(transition_rate)

        # First check that all of the symbols in sp.expr are defined (if it exists)
        if transition_rate is not None:
            for expr in sp.parse_expr(transition_rate).free_symbols:
                if str(expr) not in self.rates:
                    self.add_rate(str(expr))

        if transition_rate not in self.rates:
            self.rates.append(transition_rate)

        self.graph.add_edge(from_node, to_node, rate=transition_rate)

    def get_transition_matrix(self):
        """
        Returns a pair: labels, and the transition matrix
        """
        matrix = []
        for current_state in self.graph.nodes:
            row = []
            # Get edges incident to the state
            for incident_state in self.graph.nodes:
                if current_state == incident_state:
                    row.append(0)
                else:
                    edge = self.graph.get_edge_data(current_state, incident_state)
                    if edge is not None:
                        row.append(edge["rate"])
                    else:
                        row.append(0)
            matrix.append(row)

        matrix = sp.Matrix(matrix)
        # Compute diagonals
        n = matrix.shape[0]
        for i in range(n):
            matrix[i, i] = -sum(matrix[i, :])

        return self.graph.nodes, matrix

    def eval_transition_matrix(self, rates: dict):
        l, Q = self.get_transition_matrix()
        rates_list = [rates[rate] for rate in self.rates]
        Q_evaled = sp.lambdify(self.rates, Q)(*rates_list)
        return l, Q_evaled

    def eliminate_state_from_transition_matrix(self, labels: Union[list, None] = None):
        """eliminate_state_from_transition_matrix

        Because the state occupancy probabilities must add up to zero, the
        transition matrix is always singular. We can use this fact to remove
        one state variable from the system of equations. The labels parameter
        allows you to choose which variable is eliminated and also the ordering
        of the states.

        params:

        labels: A list of labels. This must be one less than the number of
        states in the model. The order of this list determines the ordering of
        the state variable in the outputted dynamical system.

        returns:

        Returns a pair of symbolic matrices, A & B, defining a system of ODEs of the format dX/dt = AX + B.
        """

        _, matrix = self.get_transition_matrix()
        matrix = matrix.T
        shape = sp.shape(matrix)
        assert(shape[0] == shape[1])

        if labels is None:
            labels = list(self.graph.nodes)[:-1]

        # List describing the mapping from self.graph.nodes to labels.
        # permutation[i] = j corresponds to a mapping which takes
        # graph.nodes[i] to graph.nodes[j]. Map the row to be eliminated to the
        # end.
        permutation = [labels.index(n) if n in labels else shape[0] - 1 for n in self.graph.nodes]

        matrix = matrix[permutation, permutation]

        M = sp.eye(shape[0])
        replacement_row = np.array([-1 for i in range(shape[0])])[None, :]

        M[-1, :] = replacement_row

        matrix = M @ matrix

        # Construct vector
        vec = sp.Matrix([0 for i in range(shape[0])])
        for j, el in enumerate(matrix[:, -1]):
            if el != 0:
                vec[j, 0] = el
                for i in range(shape[0]):
                    matrix[j, i] -= el
        return matrix[0:-1, 0:-1], vec[0:-1, :]

    def get_embedded_chain(self, rate_values: dict):
        _, Q = self.get_transition_matrix()
        _, Q_evaled = self.eval_transition_matrix(rate_values)

        logging.debug("Q is {}".format(Q_evaled))

        mean_waiting_times = -1 / np.diagonal(Q_evaled)

        embedded_markov_chain = np.zeros(Q_evaled.shape)
        for i, row in enumerate(Q_evaled):
            s_row = sum(row) - Q_evaled[i, i]
            for j, val in enumerate(row):
                if i == j:
                    embedded_markov_chain[i, j] = 0
                else:
                    embedded_markov_chain[i, j] = val / s_row

        logging.debug("Embedded markov chain is: {}".format(embedded_markov_chain))
        logging.debug("Waiting times are {}".format(mean_waiting_times))

        return mean_waiting_times, embedded_markov_chain

    def sample_trajectories(self, no_trajectories: int, rate_values: dict, time_range: list = [0, 1],
                            starting_distribution: Union[list, None] = None):
        no_nodes = len(self.graph.nodes)
        logging.debug("There are {} nodes".format(no_nodes))

        if starting_distribution is None:
            starting_distribution = np.around(np.array([no_trajectories] * no_nodes) / no_nodes)
            starting_distribution[0] += no_trajectories - starting_distribution.sum()

        distribution = starting_distribution

        mean_waiting_times, e_chain = self.get_embedded_chain(rate_values)

        data = [(time_range[0], *distribution)]
        culm_rows = np.zeros(e_chain.shape)
        for i in range(e_chain.shape[0]):
            sum = 0
            for j in range(e_chain.shape[1]):
                culm_rows[i, j] = e_chain[i, j] + sum
                sum += e_chain[i, j]
        t = 0
        while True:
            waiting_times = np.zeros(mean_waiting_times.shape)
            for state_index, s_i in enumerate(distribution):
                if s_i == 0:
                    waiting_times[state_index] = np.inf
                else:
                    waiting_times[state_index] = self.rng.exponential(mean_waiting_times[state_index] / (s_i))

            if t + min(waiting_times) > time_range[1] - time_range[0]:
                break

            new_t = t + min(waiting_times)
            if new_t == t:
                logging.warning("Underflow warning: timestep too small")
            t = new_t

            state_to_jump = list(waiting_times).index(min(waiting_times))

            # Find what state we will jump to
            rand = self.rng.uniform()
            jump_to = next(i for i, x in enumerate(culm_rows[state_to_jump, :]) if rand < x)

            distribution[state_to_jump] -= 1
            distribution[jump_to] += 1

            data.append((t + time_range[0], *distribution))

        df = pd.DataFrame(data, columns=['time', *self.graph.nodes])
        return df

    def get_equilibrium_distribution(self, rates: dict):
        A, B = self.eliminate_state_from_transition_matrix()
        labels = self.graph
        ss = -np.array(A.LUsolve(B).subs(rates)).astype(np.float64)
        ss = np.append(ss, 1 - ss.sum())
        return labels, ss

    def is_reversible(self):
        undirected_graph = self.graph.to_undirected(reciprocal=False, as_view=True)
        cycle_basis = nx.cycle_basis(undirected_graph)

        for cycle in cycle_basis:
            cycle.append(cycle[0])
            logging.debug("Checking cycle {}".format(cycle))

            iter = list(zip(cycle, itertools.islice(cycle, 1, None)))
            forward_rate_list = [sp.sympify(self.graph.get_edge_data(frm, to)['rate']) for frm, to in iter]
            backward_rate_list = [sp.sympify(self.graph.get_edge_data(frm, to)['rate']) for to, frm in iter]

            logging.debug("Rates moving forwards around the cycle are: %s", forward_rate_list)
            logging.debug("Rates moving backwards around the cycle are: %s", backward_rate_list)

            if None in backward_rate_list or None in forward_rate_list:
                logging.debug("Not all rates were specified.")
                return False

            forward_rate_product = sp.prod(forward_rate_list)
            backward_rate_product = sp.prod(backward_rate_list)
            if(forward_rate_product - backward_rate_product).evalf() != 0:
                return False
        return True
