import itertools
import logging
from typing import Optional, Set, Union

import myokit
import networkx as nx
import numpy as np
import pandas as pd
import pyvis
import sympy as sp
from numpy.random import default_rng


class MarkovChain():
    def __init__(self, states: Optional[list] = None, seed: Optional[int] = None, name: Optional[str] = None):

        # Initialise the graph representing the states. Each directed edge has
        # a `rate` attribute which is a string representing the transition rate
        # between the two nodes

        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.rates = set()

        self.rate_expressions = dict()

        # Initialise a random number generator for simulation. Optionally, a
        # seed can be specified.
        self.rng = default_rng(seed)
        self.name = name
        self.shared_rate_variables = []

    def mirror_model(self, prefix: str, new_rates: bool = False):
        """

        Duplicate the graph. The new nodes will be disconnected from the nodes
        in the original graph. New nodes will be the same as the original nodes
        but with the prefix prepended. This function may be used to construct
        drug trapping models.

        @params

        prefix: The prefix to prepend to the new (trapped) nodes and rates (if new_rates is True)
        new_rates: Whether or not to add a prefix to the new transition rates
        """

        trapped_graph = nx.relabel_nodes(self.graph, dict([(n, "{}{}".format(prefix, n)) for n in self.graph.nodes]))
        nx.set_node_attributes(trapped_graph, False, 'open')

        if new_rates:
            for frm, to, attr in trapped_graph.edges(data=True):
                new_rate = sp.sympify(attr['rate'])
                for symbol in new_rate.free_symbols:
                    new_symbol = prefix + str(symbol)
                    new_rate = new_rate.subs(symbol, new_symbol)

                    if new_symbol not in self.rates:
                        self.rates.add(str(new_rate))

                attr['rate'] = str(new_rate)

        new_graph = nx.compose(trapped_graph, self.graph)
        # Get open state name
        open_nodes = [n for n, d in new_graph.nodes(data=True) if d['open']]
        assert(len(open_nodes) == 1)

        self.graph = new_graph

    def add_open_trapping(self, prefix: str = "d_", new_rates: bool = False):
        """

        Construct an open trapping model by mirroring the current model and
        connecting the open state with the drugged open state

        """
        self.mirror_model(prefix, new_rates)
        self.add_rates(("drug_on", "drug_off"))
        open_nodes = [n for n, d in self.graph.nodes(data=True) if d['open']]
        self.add_both_transitions(open_nodes[0], "d_{}".format(open_nodes[0]), 'drug_on', 'drug_off')

    def add_state(self, label, open: bool = False):
        """

        Add a new state to the model

        @params:

        label: The label to attach to the new state open: A flag determining
        whether or not the new node has the `open` attribute. This is used to
        compute the current flowing through the channel

        """
        label = sp.sympify(label)

        if not isinstance(label, sp.core.expr.Expr):
            raise Exception(f'{label} is not a valid sympy expression')

        if len(label.free_symbols) == 1:
            self.graph.add_node(str(label), open=open)
        else:
            raise Exception(f'{label} is not a valid state label.')

    def add_states(self, states: list):
        """

        Adds a list of states to the model.

        @params

        states: A list where each element is either a string specifying the
        name of the new label or a pair of values -- the new label and a flag
        determining the `open` attribute. If the `open` flag is omitted, the
        new node will have `open=False`

        """
        for state in states:
            if isinstance(state, str):
                self.add_state(state)
            else:
                self.add_state(*state)

    def add_rate(self, rate: str):
        """

        Add a new transition rate to the model. These are stored in self.rates.

        @param

        rate: A string defining the rate to be added

        """

        # Check that the new rate isn't some complicated sympy expression
        # TODO test this and add nice exception

        symrate = sp.sympify(rate)
        if len(symrate.atoms()) != 1:
            raise Exception()

        if rate in self.rates:
            # TODO
            raise Exception()
        else:
            self.rates.add(rate)

    def add_rates(self, rates: list):
        """

        Add a list of rates to the model

        @param

        rates: A list of strings to be added to self.rates

        """
        for rate in rates:
            self.add_rate(rate)

    def add_transition(self, from_node: str, to_node: str, transition_rate: Optional[str],
                       label: Optional[str] = None):
        """

        Adds an edge describing the transition rate between `from_node` and `to_node`.

        @params

        from_node: The state that the transition rate is incident from

        to_node: The state that the transition rate is incident to

        transition rate: A string identifying this transition with a rate from
        self.rates. If rate is not present in self.rates and exception will be
        thrown.

        """

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
            self.rates.add(transition_rate)

        if label is None:
            label = transition_rate
        self.graph.add_edge(from_node, to_node, rate=transition_rate, label=label)

    def add_both_transitions(self, frm: str, to: str, fwd_rate: Union[str, sp.Expr, None], bwd_rate: Optional[str]):
        """A helper function to add forwards and backwards rates between two
        states. This is a convenient way to connect new states to the model.

        @params

        frm, to: the states to be connect. These must already be in
        self.graph.nodes or an exception will be thrown.

        fwd_rate: The transition rate from `frm` to `to`

        bwd_rate: The transition rate from `to` to `frm`

        """
        self.add_transition(frm, to, fwd_rate)
        self.add_transition(to, frm, bwd_rate)

    def get_transition_matrix(self, use_parameters: bool = False):
        """Computes a matrix where the off-diagonals describe the transition rates
        between states. This matrix is the Q matrix of the Markov Chain.

        use_parameters: If true substitute in parameters of the transition rates

        @Returns

        a 2-tuple: labels, and the transition matrix such that the labels give
        the states that each column corresponds to.

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

        if use_parameters:
            if len(self.rate_expressions) == 0:
                raise Exception()
            else:
                matrix = matrix.subs(self.rate_expressions)

        return list(self.graph.nodes), matrix

    def eval_transition_matrix(self, rates: dict):
        """
        Evaluate the transition matrix given values for each of the transition rates

        @params

        rates: A dictionary defining the value of each transition rate e.g rates['K1'] = 1.
        """
        l, Q = self.get_transition_matrix()
        rates_list = [rates[rate] for rate in self.rates]
        Q_evaled = sp.lambdify(list(self.rates), Q)(*rates_list)
        return l, Q_evaled

    def eliminate_state_from_transition_matrix(self, labels: Optional[list] = None, use_parameters: bool = False):
        """eliminate_state_from_transition_matrix

        Because the state occupancy probabilities must add up to zero, the
        transition matrix is always singular. We can use this fact to remove
        one state variable from the system of equations. The labels parameter
        allows you to choose which variable is eliminated and also the ordering
        of the states.

        @params

        labels: A list of labels. This must be one less than the number of
        states in the model. The order of this list determines the ordering of
        the state variable in the outputted dynamical system.

        use_parameters: If true substitute in parameters of the transition rates

        @returns

        Returns a pair of symbolic matrices, A & B, defining a system of ODEs of the format dX/dt = AX + B.
        """

        for label in labels:
            if label not in self.graph.nodes():
                raise Exception()

        _, matrix = self.get_transition_matrix()
        matrix = matrix.T
        shape = sp.shape(matrix)
        assert shape[0] == shape[1]

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

        if use_parameters:
            if len(self.rate_expressions) == 0:
                raise Exception()
            matrix = matrix.subs(self.rate_expressions)

        return matrix[0:-1, 0:-1], vec[0:-1, :]

    def get_embedded_chain(self, rate_values: dict):
        """Compute the embedded DTMC and associated waiting times
        given values for each of the transition rates

        @param

        rates: A dictionary defining the value of each transition rate e.g rates['K1'] = 1.

        @returns

        A 3-tuple: `labs` describes the order states in the returned
        values, `mean_wainting_times` is a numpy array describing the mean
        waiting times for each state, `embedded_markov_chain` is a numpy array
        describing the embedded DTMC of this Markov chain

        """

        labs, Q = self.get_transition_matrix()
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

        return labs, mean_waiting_times, embedded_markov_chain

    def sample_trajectories(self, no_trajectories: int, rate_values: dict, time_range: list = [0, 1],
                            starting_distribution: Optional[list] = None):
        """Samples trajectories of the Markov chain using a Gillespie algorithm.

        @params no_trajectories: The number of simulations to run (number of
        channels) rate_values: A dictionary defining the (constant) value of
        each transition rate time_range: A range of times durig which to
        simulate the model starting_distribution: Defines the number of
        trajectories starting in each state. If this is None, we default to
        starting with roughly the same number of channels in each state.

        @returns

        A pandas dataframe describing the number of channels in each state for
        the times in time_range

        """

        no_nodes = len(self.graph.nodes)
        logging.debug('There are {} nodes'.format(no_nodes))

        if starting_distribution is None:
            starting_distribution = np.around(np.array([no_trajectories] * no_nodes) / no_nodes)
            starting_distribution[0] += no_trajectories - starting_distribution.sum()

        distribution = starting_distribution

        labels, mean_waiting_times, e_chain = self.get_embedded_chain(rate_values)

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
        """Compute the equilibrium distribution of the CTMC for the specified transition rate values

        @params

        rates: A dictionary specifying the values of each transition rate

        @returns

        A 2-tuple, ss describesthe equilibrium distribution and labels defines
        which entry relates to which state

        """
        A, B = self.eliminate_state_from_transition_matrix()
        labels = self.graph
        ss = -np.array(A.LUsolve(B).subs(rates)).astype(np.float64)
        ss = np.append(ss, 1 - ss.sum())
        return labels, ss

    def is_reversible(self):
        """Checks whether or not the Markov chain is reversible for any set of non-zero
        transition rate values. This method does not check for specific
        transition rate values. We assume that all transition rates are
        non-zero and follow Colquhoun et al. (2004) https://doi.org/10.1529/biophysj.103.038679


        @returns

        True if and only if the Markov chain is reversible for every
        possible combination of transition rate variables

        """

        # Digraph must be strongly connected in order for the chain to be
        # reversible. In other words it must be possible to transition from any
        # state to any other state in some finite number of transitions
        if not nx.algorithms.components.is_strongly_connected(self.graph):
            return False

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

    def draw_graph(self, filepath: Optional[str] = None, show_options: bool =
                   False, show_rates: bool = False, show_parameters: bool = False):
        """Visualise the graph as a webpage using pyvis.

        @params

        filepath: An optional filepath to save the file to. If this is
        None, the graph won't be saved and will be opened in a web browser (if possible)

        show_options: Whether or not the options menu should be displayed on the webpage

        show_parameters: Whether or not we should display the transition rates instead of their labels

        show_parameters: Whether or not we should show the parameterised version of each transition rate

        """
        for _, _, data in self.graph.edges(data=True):
            if 'label' not in data or show_rates:
                data['label'] = data['rate']
            elif show_parameters:
                if len(self.rate_expressions) == 0:
                    raise Exception()
                else:
                    data['label'] = str(self.rate_expressions[data['rate']])

        nt = pyvis.network.Network(directed=True)
        nt.from_nx(self.graph)
        nt.set_edge_smooth('dynamic')
        if show_options:
            nt.show_buttons()
        if filepath is not None:
            nt.save_graph(filepath)
        else:
            nt.show('Markov_builder_graph.html')

    def substitute_rates(self, rates_dict: dict):
        """Substitute expressions for into the transition rates.

        This function modifies the `rate` attribute of edges in self.graph

        @param: rates_dict: A dictionary where each key must be in self.rates
        and the value is the corresponding expression to be substituted in.

        """

        for rate in rates_dict:
            if rate not in self.rates:
                raise Exception()
        for _, _, d in self.graph.edges(data=True):
            if d['rate'] in rates_dict:
                if 'label' not in d:
                    d['label'] = d['rate']
                d['rate'] = str(sp.sympify(d['rate']).subs(rates_dict))

    def parameterise_rates(self, rate_dict: dict, shared_variables: list = []) -> None:
        """Define a set of parameters for the transition rates.

        Parameters declared as
        'dummy variables' are relabelled and the expressions stored in
        self.rate_expressions. This results in a parameterisation of the whole
        model. The most common choice is to use an expression of the form k =
        exp(a + b*V) or k = exp(a - b*V) where a and b are dummy variables and
        V is the membrane voltage (a variable shared between transition rates).

        @params
        rate_dict: A dictionary containing a key for each rate in self.rates
        with corresponding tuples defining an expression for the rate and a
        list of relevant dummy variables e.g {k: ('exp(e + bV)', (a,b))}.

        shared_variables: A list of variables that may be shared between
        transition rates (such as the membrane voltage). All free symbols in
        each expression which are not dummy_variables must be defined as
        shared_variables.

        TODO Exception messages

        """

        # Check that shared_variables is a list (and not a string!)
        if isinstance(shared_variables, str):
            raise TypeError("shared_variables is a string but must be a list")

        # Validate rate dictionary
        for r in rate_dict:
            if r not in self.rates:
                raise Exception()

        rate_expressions = dict()
        param_counter = 0
        for r in self.rates:
            if r in rate_dict:
                expression, dummy_variables = rate_dict[r]
                expression = sp.sympify(expression)
                for symbol in expression.free_symbols:
                    variables = list(dummy_variables) + list(shared_variables)
                    if str(symbol) not in variables:
                        raise Exception(
                            f"Symbol, {symbol} was not found in dummy variables or shared_variables, {variables}.")
                subs_dict = {u: f"p_{i + param_counter}" for i, u in enumerate(dummy_variables)}
                param_counter += len(dummy_variables)
                rate_expressions[r] = sp.sympify(expression).subs(subs_dict)

        self.rate_expressions = rate_expressions
        self.shared_rate_variables = shared_variables

    def get_parameter_list(self) -> Set[str]:
        """
        Returns a list of strings corresponding the symbols in self.rate_expressions.
        """

        rates = set()

        for r in self.rate_expressions:
            for symbol in self.rate_expressions[r].free_symbols:
                rates.add(str(symbol))

        return sorted(rates)

    def get_myokit_model(self, name: str = "", voltage: str = 'V'):

        if name == "":
            name = self.name

        model = myokit.Model(name)

        model.add_component('markov_chain')
        comp = model['markov_chain']

        states, Q = self.get_transition_matrix()
        d_equations = dict(zip(states, sp.Matrix(states).T * Q))

        # Add required time and pace variables
        model.add_component('engine')
        model['engine'].add_variable('time')
        model['engine'].add_variable('pace')

        model['engine']['time'].set_binding('time')
        model['engine']['time'].set_rhs(0)

        model['engine']['pace'].set_binding('pace')
        model['engine']['pace'].set_rhs(0)

        # Add parameters to the model
        for parameter in self.get_parameter_list():
            if parameter == voltage:
                model.add_component('membrane')
                model['membrane'].add_variable('V')
                model['membrane']['V'].set_rhs('engine.pace')
                model['membrane']['V'].set_rhs(0)
                comp.add_alias(voltage, model['membrane']['V'])
            else:
                comp.add_variable(parameter)

        for rate in self.rates:
            comp.add_variable(rate)
            if rate in self.rate_expressions:
                expr = self.rate_expressions[rate]
                comp[rate].set_rhs(str(expr))

        for state in self.graph.nodes():
            comp.add_variable(state)
            var = comp[state]
            var.promote()

        for state in self.graph.nodes():
            comp[state].set_rhs(str(d_equations[state]))

        return model
