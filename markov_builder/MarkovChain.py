import itertools
import logging
from dataclasses import asdict, dataclass, is_dataclass
from typing import List, Optional, Tuple

import myokit
import networkx as nx
import numpy as np
import pandas as pd
import pyvis
import sympy as sp
from numpy.random import default_rng


@dataclass
class MarkovStateAttributes:
    """A dataclass defining what attributes each state in a MarkovChain should have, and their default values"""
    open_state: bool = False
    inactive: bool = False


class MarkovChain():
    """A class used to construct continuous time Markov Chains/compartmental models using networkx.

    Various helper functions are included to generate equations, code for the
    Markov chains and to test whether the model has certain properties.
    """

    def __init__(self, states: Optional[list] = None, state_attributes_class:
                 Optional[MarkovStateAttributes] = None, seed: Optional[int] = None, name:
                 Optional[str] = None):

        # Initialise the graph representing the states. Each directed edge has
        # a `rate` attribute which is a string representing the transition rate
        # between the two nodes

        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.rates = set()

        # Initialise a random number generator for simulation. Optionally, a
        # seed can be specified.
        self.rng = default_rng(seed)
        self.name = name
        self.shared_rate_variables = set()

        self.rate_expressions = {}
        self.default_values = {}

        self.auxiliary_expression = None

        if state_attributes_class is None:
            state_attributes_class = MarkovStateAttributes

        self.state_attributes_class = state_attributes_class
        self.reserved_names = []
        self.auxiliary_variable = 'auxiliary_expression'

        if not is_dataclass(self.state_attributes_class):
            raise Exception("state_attirbutes_class must be a dataclass")

    def mirror_model(self, prefix: str, new_rates: bool = False) -> None:
        """ Duplicate all states and rates in the model such that there are two identical components.

        The new nodes will be disconnected from the nodes in the original graph.
        New nodes will be the same as the original nodes but with the prefix
        prepended. This function may be used to construct drug trapping models.

        :param prefix: The prefix to prepend to the new (trapped) nodes and rates (if new_rates is True)
        :param new_rates: Whether or not to add a prefix to the new transition rates

       """

        trapped_graph = nx.relabel_nodes(self.graph, dict([(n, "{}{}".format(prefix, n)) for n in self.graph.nodes]))
        nx.set_node_attributes(trapped_graph, False, 'open_state')

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
        open_nodes = [n for n, d in new_graph.nodes(data=True) if d['open_state']]
        assert(len(open_nodes) == 1)

        self.graph = new_graph

    def add_open_trapping(self, prefix: str = "d_", new_rates: bool = False) -> None:
        """Construct an open trapping model by mirroring the current model and connecting the open states.

        :param prefix: The prefix to prepend onto the new (trapped) nodes and rates (if new_rates is true.)
        :param new_rates: Whether or not to create new transition rates for the new mirrored edges

        """
        self.mirror_model(prefix, new_rates)
        self.add_rates(("drug_on", "drug_off"))
        open_nodes = [n for n, d in self.graph.nodes(data=True) if d['open_state']]
        self.add_both_transitions(open_nodes[0], "d_{}".format(open_nodes[0]), 'drug_on', 'drug_off')

    def add_state(self, label, **kwargs) -> None:
        """Add a new state to the model.

        :param label: The label to attach to the new state open.

        :param kwargs: Keyword arguments to use with graph.add_node specifying additional attributes for the new node.

        """
        label_symbol = sp.sympify(label)

        attributes = asdict(self.state_attributes_class(**kwargs))

        if label in self.reserved_names:
            raise Exception("label %s is reserved", label)

        if not isinstance(label_symbol, sp.core.expr.Expr):
            raise Exception(f'{label} is not a valid sympy expression')

        if not len(label_symbol.free_symbols) == 1:
            raise Exception(f'{label} is not a valid state label.')

        self.graph.add_node(str(label), **attributes)

    def add_states(self, states: list) -> None:
        """Adds a list of states to the model.

        :param states: A list specifying the name of the new label and a dictionary of attributes for new node.
        """
        for state in states:
            if isinstance(state, str):
                attr = {}
            else:
                attr = state[-1]
                state = state[0]
            self.add_state(state, **attr)

    def add_rate(self, rate: str) -> None:
        """

        Add a new transition rate to the model. These are stored in self.rates.

        :param rate: A string defining the rate to be added


        """

        # Check that the new rate isn't some complicated sympy expression
        # TODO test this and add nice exception

        if rate in self.reserved_names:
            raise Exception('Name %s is reserved', rate)

        symrate = sp.sympify(rate)
        if len(symrate.atoms()) != 1:
            raise Exception()

        if rate in self.rates:
            # TODO
            raise Exception()
        else:
            self.rates.add(rate)

    def add_rates(self, rates: list) -> None:
        """
        Add a list of rates to the model

        :param rates: A list of strings to be added to self.rates

        """
        for rate in rates:
            self.add_rate(rate)

    def add_transition(self, from_node: str, to_node: str, transition_rate: Optional[str],
                       label: Optional[str] = None, update=False) -> None:
        """Adds an edge describing the transition rate between `from_node` and `to_node`.

        :param from_node: The state that the transition rate is incident from
        :param to_node: The state that the transition rate is incident to
        :param transition rate: A string identifying this transition with a rate from self.rates.
        :param update: If false and exception will be thrown if an edge between from_node and to_node already exists
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
        if (from_node, to_node) in self.graph.edges():
            if update:
                self.graph.add_edge(from_node, to_node, rate=transition_rate, label=label)
            else:
                raise Exception(f"An edge already exists between {from_node} and {to_node}. \
                Edges are {self.graph.edges()}")
        else:
            self.graph.add_edge(from_node, to_node, rate=transition_rate, label=label)

    def add_both_transitions(self, frm: str, to: str, fwd_rate: str = None,
                             bwd_rate: str = None, update=True) -> None:
        """A helper function to add forwards and backwards rates between two
        states.

        This is a convenient way to connect new states to the model.

        :param frm: Either of the two states to be connected.
        :param to: Either of the two states to be connected.
        :param fwd_rate: The transition rate from `frm` to `to`.
        :param bwd_rate: The transition rate from `to` to `frm`.
        :param update: If false and exception will be thrown if an edge between from_node and to_node already exists
        """

        self.add_transition(frm, to, fwd_rate, update=update)
        self.add_transition(to, frm, bwd_rate, update=update)

    def get_transition_matrix(self, use_parameters: bool = False) -> Tuple[List[str], sp.Matrix]:
        """Compute the Q Matrix of the Markov chain. Q[i,j] is the transition rate between states i and j.

        :param use_parameters: If true substitute in parameters of the transition rates
        :return: a 2-tuple, labels, and the transition matrix such that the labels column correspond.

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
            matrix = matrix.subs(self.rate_expressions)

        return list(self.graph.nodes), matrix

    def eval_transition_matrix(self, rates_dict: dict) -> Tuple[List[str], sp.Matrix]:
        """
        Evaluate the transition matrix given values for each of the transition rates.

        :param rates: A dictionary defining the value of each transition rate e.g rates['K1'] = 1.
        """
        l, Q = self.get_transition_matrix(use_parameters=True)
        Q_evaled = np.array(Q.evalf(subs=rates_dict))
        return l, Q_evaled

    def eliminate_state_from_transition_matrix(self, labels: Optional[list] = None,
                                               use_parameters: bool = False) -> Tuple[sp.Matrix, sp.Matrix]:
        """Returns a matrix, A, and vector, B, corresponding to a linear ODE system describing the state probabilities.

        Because the state occupancy probabilities must add up to zero, the
        transition matrix is always singular. We can use this fact to remove
        one state variable from the system of equations. The labels parameter
        allows you to choose which variable is eliminated and also the ordering
        of the states.

        :param labels: A list of labels. The order of which determines the ordering of outputted system.
        :param use_parameters: If true substitute in parameters of the transition rates
        :return: A pair of symbolic matrices, A & B, defining a system of ODEs of the format dX/dt = AX + B.
        """

        if labels is None:
            labels = list(self.graph.nodes)[:-1]

        for label in labels:
            if label not in self.graph.nodes():
                raise Exception()

        _, matrix = self.get_transition_matrix()
        matrix = matrix.T
        shape = sp.shape(matrix)
        assert shape[0] == shape[1]

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
            else:
                matrix = matrix.subs(self.rate_expressions)
                vec = vec.subs(self.rate_expressions)

        return matrix[0:-1, 0:-1], vec[0:-1, :]

    def get_embedded_chain(self, param_dict: dict = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Compute the embedded DTMC and associated waiting times given values for each of the transition rates

        :param rates: A dictionary defining the value of each transition rate e.g rates['K1'] = 1.

        :return: 3-tuple: the state labels, the waiting times for each state, and the embedded Markov chain.

        """

        if param_dict is None:
            param_dict = self.default_values

        labs, Q = self.get_transition_matrix()
        _, Q_evaled = self.eval_transition_matrix(param_dict)

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

    def sample_trajectories(self, no_trajectories: int, time_range: list = [0, 1],
                            param_dict: dict = None,
                            starting_distribution: list = None) -> pd.DataFrame:
        """Samples trajectories of the Markov chain using a Gillespie algorithm.

        :param no_trajectories: The number of simulations to run (number of channels)
        :param time_range: A range of times durig which to simulate the model
        :param param_dict: A dictionary defining the (constant) value of each transition rate
        :param starting_distribution: The number of samples starting in each state. Defaults to an even distribution.
        :return: A pandas dataframe describing the number of channels in each state for the times in time_range

        """

        no_nodes = len(self.graph.nodes)
        logging.debug(f"There are {no_nodes} nodes")

        if param_dict is not None:
            param_list = self.get_parameter_list()
            # default missing values to those in self.default_values
            param_dict = {param: param_dict[param]
                          if param in param_dict
                          else self.default_values[param]
                          for param in param_list}
        else:
            param_dict = self.default_values

        if starting_distribution is None:
            starting_distribution = np.around(np.array([no_trajectories] * no_nodes) / no_nodes)
            starting_distribution[0] += no_trajectories - starting_distribution.sum()

        distribution = starting_distribution

        labels, mean_waiting_times, e_chain = self.get_embedded_chain(param_dict)

        data = [(time_range[0], *distribution)]

        cumul_rows = np.array(list(map(np.cumsum, e_chain)))

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
            jump_to = next(i for i, x in enumerate(cumul_rows[state_to_jump, :]) if rand < x)

            distribution[state_to_jump] -= 1
            distribution[jump_to] += 1

            data.append((t + time_range[0], *distribution))

        df = pd.DataFrame(data, columns=['time', *self.graph.nodes])
        return df

    def get_equilibrium_distribution(self, param_dict: dict = None) -> Tuple[List[str], np.array]:
        """Compute the equilibrium distribution of the CTMC for the specified transition rate values

        :param param_dict: A dictionary specifying the values of each transition rate

        :return: A 2-tuple describing equilibrium distribution and labels defines which entry relates to which state

        """

        if param_dict is not None:
            param_list = self.get_parameter_list()
            # default missing values to those in self.default_values
            param_dict = {param: param_dict[param]
                          if param in param_dict
                          else self.default_values[param]
                          for param in param_list}
        else:
            param_dict = self.default_values

        A, B = self.eliminate_state_from_transition_matrix(use_parameters=True)

        labels = self.graph.nodes()
        ss = -np.array(A.LUsolve(B).evalf(subs=param_dict))
        logging.debug("ss is %s", ss)
        ss = np.append(ss, 1 - ss.sum())
        return labels, ss

    def is_reversible(self) -> bool:
        """Checks symbolically whether or not the Markov chain is reversible for any set of non-zero transition rate values.

        We assume that all transition rates are always non-zero and follow
        Colquhoun et al. (2004) https://doi.org/10.1529/biophysj.103.

        :return: A bool which is true if Markov chain is reversible (assuming non-zero transition rates).

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

    def draw_graph(self, filepath: str = None, show_options: bool =
                   False, show_rates: bool = False, show_parameters: bool = False):
        """Visualise the graph as a webpage using pyvis.

        :param filepath: An optional filepath to save the file to. If this is None, will be opened as a webpage instead.
        :param show_options: Whether or not the options menu should be displayed on the webpage
        :param show_parameters: Whether or not we should display the transition rates instead of their labels
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
        """Substitute expressions into the transition rates.

        This function modifies the `rate` attribute of edges in self.graph

        :param rates_dict: A dictionary of rates and their corresponding expressions.

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

        :param rate_dict: A dictionary with a 2-tuple containing an expression and dummy variables for each rate.
        :param shared_variables: A list of variables that may be shared between transition rates

        """

        # Check that shared_variables is a list (and not a string!)
        if isinstance(shared_variables, str):
            raise TypeError("shared_variables is a string but must be a list")

        for v in shared_variables:
            if v in self.reserved_names:
                raise Exception('name %s is reserved', v)

        # Validate rate dictionary
        for r in rate_dict:
            if r not in self.rates:
                raise Exception(f"Tried to parameterise rate {r} but it was not present in the model")

        rate_expressions = {}
        default_values_dict = {}
        for r in self.rates:
            if r in rate_dict:
                if len(rate_dict[r]) == 2:
                    expression, dummy_variables = rate_dict[r]
                    default_values = []
                else:
                    expression, dummy_variables, default_values = rate_dict[r]
                if len(dummy_variables) < len(default_values):
                    raise ValueError("dummy variables list and default values list have mismatching lengths.\
                    Lengths {} and {}".format(len(dummy_variables), len(default_values)))

                expression = sp.sympify(expression)

                for symbol in expression.free_symbols:
                    variables = list(dummy_variables) + list(shared_variables)
                    if str(symbol) not in variables:
                        raise Exception(
                            f"Symbol, {symbol} was not found in dummy variables or shared_variables, {variables}.")
                subs_dict = {u: f"{r}_{u}" for i, u in enumerate(dummy_variables)}
                rate_expressions[r] = sp.sympify(expression).subs(subs_dict)

                # Add default values to dictionary
                for u, v in zip(dummy_variables, default_values):
                    new_var_name = f"{r}_{u}"
                    if new_var_name in default_values_dict:
                        raise Exception(f"A parameter with label {new_var_name} is already present in the model.")
                    default_values_dict[new_var_name] = v

        self.rate_expressions = {**self.rate_expressions, **rate_expressions}
        self.default_values = {**self.default_values, **default_values_dict}

        self.shared_rate_variables = self.shared_rate_variables.union(shared_variables)

    def get_parameter_list(self) -> List[str]:
        """
        Get a list describing every parameter in the model

        :return: a list of strings corresponding the symbols in self.rate_expressions and self.shared_rate_variables.
        """

        rates = set()

        for r in self.rate_expressions:
            for symbol in self.rate_expressions[r].free_symbols:
                rates.add(str(symbol))

        rates = rates.union([str(sym) for sym in self.shared_rate_variables])

        return sorted(rates)

    def generate_myokit_model(self, name: str = "",
                              membrane_potential: str = 'V',
                              drug_binding=False, eliminate_state=None) -> myokit.Model:
        """Generate a myokit Model instance describing this Markov model.

        Build a myokit model from this Markov chain using the parameterisation
        defined by self.rate_expressions. If a rate does not have an entry in
        self.rate_expressions, it is treated as a constant.

        All initial conditions and parameter values should be set before the
        model is run.

        :param name: A name to give to the model. Defaults to self.name.

        :param membrane_voltage: A label defining which variable should be treated as the membrane potential.

        :param eliminate_rate: Which rate (if any) to eliminate in order to reduce the number of ODEs in the system.

        :return: A myokit.Model built using self

        """

        if name == "":
            name = self.name

        model = myokit.Model(name)

        model.add_component('markov_chain')
        comp = model['markov_chain']

        if eliminate_state is not None:
            states = [state for state in self.graph.nodes()
                      if state != eliminate_state]

            A, B = self.eliminate_state_from_transition_matrix(states)
            d_equations = dict(zip(states, A @ sp.Matrix(states) + B))

        else:
            states, Q = self.get_transition_matrix()
            d_equations = dict(zip(states, sp.Matrix(states).T @ Q))

        # Add required time and pace variables
        model.add_component('engine')
        model['engine'].add_variable('time')

        model['engine']['time'].set_binding('time')
        model['engine']['time'].set_rhs(0)

        drug_concentration = 'D' if drug_binding else None

        # Add parameters to the model
        for parameter in self.get_parameter_list():
            if parameter == membrane_potential:
                model.add_component('membrane')
                model['membrane'].add_variable('V')
                model['membrane']['V'].set_rhs(0)
                model['membrane']['V'].set_binding('pace')
                comp.add_alias(membrane_potential, model['membrane']['V'])
            elif drug_binding and parameter == drug_concentration:
                model.add_component('drug')
                model['drug'].add_variable('D')
                model['drug']['D'].set_rhs(0)
                comp.add_alias(drug_concentration, model['drug']['D'])
            else:
                comp.add_variable(parameter)
                if parameter in self.default_values:
                    comp[parameter].set_rhs(self.default_values[parameter])

        for rate in self.rates:
            comp.add_variable(rate)
            if rate in self.rate_expressions:
                expr = self.rate_expressions[rate]
                comp[rate].set_rhs(str(expr))

        for state in self.graph.nodes():
            comp.add_variable(state)

        for state in states:
            var = comp[state]
            var.promote()
            var.set_rhs(str(d_equations[state]))
            var.set_state_value(0)

        comp[states[-1]].set_state_value(1)

        if eliminate_state is not None:
            rhs_str = "1 "
            for state in states:
                rhs_str += f"-{state}"

            comp[eliminate_state].set_rhs(rhs_str)

        if self.auxiliary_expression is not None:
            comp.add_variable(self.auxiliary_variable)
            comp[self.auxiliary_variable].set_rhs(str(self.auxiliary_expression))
        return model

    def define_auxiliary_expression(self, expression: sp.Expr, label: str = None, default_values: dict = {}) -> None:
        """Define an auxiliary output variable for the model.

        :param expression: A sympy expression defining the auxiliary variable
        :param label: A str naming the variable e.g IKr
        :param default_values: A dictionary of the default values of any parameter used in the auxiliary expression

        """
        if label in self.graph.nodes() or label in self.reserved_names:
            raise Exception('Name %s not available', label)
        else:
            self.reserved_names = self.reserved_names + [label]

        self.auxiliary_variable = label

        if not isinstance(expression, sp.Expr):
            raise Exception()

        for symbol in expression.free_symbols:
            if str(symbol) not in self.graph.nodes():
                if self.shared_rate_variables is None:
                    self.shared_rate_variables = set(str(symbol))
                else:
                    self.shared_rate_variables.add(str(symbol))

        for symbol in default_values:
            symbol = sp.sympify(symbol)
            if symbol not in expression.free_symbols:
                raise Exception()
            if symbol in self.default_values:
                raise Exception()

        self.default_values = {**self.default_values, **default_values}
        self.auxiliary_expression = expression

    def as_latex(self, state_to_remove: str = None, include_auxiliary_expression: bool = False) -> str:
        """Creates a LaTeX expression describing the Markov chain, its parameters and
        optionally, the auxiliary equation

        :param state_to_remove: The name of the state (if any) to eliminate from the system.

        :param include_auxiliary_expression: Whether or not to include the auxiliary expression in the output

        :returns: A python string containing the relevant LaTeX code.

        """

        if state_to_remove is None:
            # Get Q matrix
            labels, Q = self.get_transition_matrix()
            Q_matrix_str = str(sp.latex(Q))
            eqn = "\\begin{equation}\\dfrac{dX}{dt} = " + Q_matrix_str + "X \\end{equation}"

            X_defn = "\\begin{equation}" + sp.latex(sp.Matrix(self.graph.nodes()))\
                + "\\end{equation}"

        else:
            if state_to_remove not in map(str, self.graph.nodes()):
                raise Exception("%s not in model", state_to_remove)
            labels = [label for label in self.graph.nodes()
                      if str(label) != str(state_to_remove)]

            if len(labels) != len(self.graph.nodes()) - 1:
                raise Exception("model has duplicated states %s",
                                self.graph.nodes())

            A, B = self.eliminate_state_from_transition_matrix(labels)

            eqn = "\\begin{equation}\\dfrac{dX}{dt} = " + sp.latex(A) + "X"\
                + " + " + sp.latex(B) + "\\end{equation}"
            X_defn = "\\begin{equation}" + sp.latex(sp.Matrix(labels)) \
                + "\\end{equation}\n"

        if len(self.rate_expressions) > 0:
            eqns = ",\\\\ \n".join([f"{sp.latex(sp.sympify(rate))} &= {sp.latex(expr)}" for rate, expr
                                    in self.rate_expressions.items()])
            eqns += ','

            rate_definitions = "\\begin{align}" + eqns + "\\end{align} \n\n"

            return_str = f"{eqn}\n where {rate_definitions} and {X_defn}"
        else:
            return_str = f"{eqn} where {X_defn}"

        if include_auxiliary_expression:
            if self.auxiliary_expression is None:
                raise Exception("No auxiliary expression present in the MarkovChain")
            return_str = "\\begin{equation}" + sp.latex(self.auxiliary_expression) + \
                "\\end{equation}" + "\n where" + return_str
        return return_str
