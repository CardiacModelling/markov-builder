#!/usr/bin/env python3

import itertools
import logging
import os
import unittest

import matplotlib
import matplotlib.pyplot as plt
import myokit
import myokit as mk
import networkx as nx
import numpy as np
import sympy as sp

from markov_builder.models.thirty_models import (
    model_00,
    model_01,
    model_02,
    model_03,
    model_04,
    model_05,
    model_06,
    model_07,
    model_08,
    model_09,
    model_10,
    model_11,
    model_12,
    model_13,
    model_14,
    model_30,
)

matplotlib.use('pdf')


class TestThirtyModels(unittest.TestCase):

    def setUp(self):
        """This is run by unittest before the tests in this class are performed.

        Create an output directory (if it doesn't already exist). An
        alternative output path can be used by setting the
        MARKOVBUILDER_TEST_OUTPUT environment variable

        """
        test_output_dir = os.environ.get('MARKOVBUILDER_TEST_OUTPUT', os.path.join(
            'test_output', self.__class__.__name__))
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

        self.models = [
            model_00, model_01, model_02, model_03, model_04,
            model_05, model_06, model_07, model_08, model_09, model_10,
            model_11, model_12, model_13, model_14, model_30
        ]

        self.disconnected_models = [model_03, model_09, model_10]

        self.model_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                              30]

    def test_generate_myokit(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"initiating {name}")
            mc = model()
            myokit_model = mc.generate_myokit_model()
            myokit.save(os.path.join(self.output_dir, f"{model.__name__}.mmt"), myokit_model)
        return

    def test_visualise_graphs(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"initiating {name}")
            mc = model()

            mc.draw_graph(os.path.join(self.output_dir, f"{name}_graph.html"),
                          show_parameters=False)
            mc.draw_graph(os.path.join(self.output_dir, f"{name}_graph_with_parameters.html"),
                          show_parameters=True)

            nx.drawing.nx_agraph.write_dot(mc.graph, os.path.join(self.output_dir,
                                                                  "%s_dotfile.dot" % name))

    def test_connected(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"initiating {name}")

            mc = model()
            self.assertTrue(mc.is_connected() ^ (model in self.disconnected_models))

    def test_reversible(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"initiating {name}")

            if model in self.disconnected_models:
                continue

            mc = model()
            if not mc.is_reversible():
                undirected_graph = mc.graph.to_undirected(reciprocal=False, as_view=True)
                cycle_basis = nx.cycle_basis(undirected_graph)

                for cycle in cycle_basis:
                    iterator = list(zip(cycle, itertools.islice(cycle, 1, None)))
                    forward_rate_list = [sp.sympify(mc.graph.get_edge_data(frm, to)['rate']) for frm, to in iterator]
                    backward_rate_list = [sp.sympify(mc.graph.get_edge_data(frm, to)['rate']) for to, frm in iterator]

                    logging.debug(mc.rate_expressions)

                    # Substitute in expressions
                    forward_rate_list = [rate.subs(mc.rate_expressions) for
                                         rate in forward_rate_list]
                    backward_rate_list = [rate.subs(mc.rate_expressions) for rate in
                                          backward_rate_list]

                    forward_rate_product = sp.prod(forward_rate_list)
                    backward_rate_product = sp.prod(backward_rate_list)
                    if (forward_rate_product - backward_rate_product).evalf() != 0:
                        logging.error("Rates moving forwards around the cycle are: %s", forward_rate_list)
                        logging.error("Rates moving backwards around the cycle are: %s", backward_rate_list)

            self.assertTrue(mc.is_reversible())

    def test_myokit_simulation_output(self):
        mmt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models_myokit')
        Erev = -88

        comparison_plot_dir = os.path.join(self.output_dir, 'comparison_plots')

        if not os.path.exists(comparison_plot_dir):
            os.makedirs(comparison_plot_dir)

        for i, model in zip(self.model_indices, self.models):
            if i is None or model is None:
                continue
            name = model.__name__
            logging.debug(f"initiating {name}")

            mc = model()
            mk_protocol_filename = os.path.join(mmt_dir,
                                                'simplified-staircase.mmt')
            mk_protocol = mk.load_protocol(mk_protocol_filename)

            mk_model = mk.load_model(os.path.join(mmt_dir,
                                                  f"model-{i}.mmt"))

            sim = mk.Simulation(mk_model, mk_protocol)
            sim.set_tolerance(1e-9, 1e-9)
            sim.set_constant('nernst.EK', Erev)

            tmax = mk_protocol.characteristic_time()
            times = np.linspace(0, tmax, int(tmax) + 1)
            sim.pre(5000)

            log = sim.run(tmax, log_times=times, log=['ikr.IKr'])

            mk_IKr = np.array(log['ikr.IKr'])

            generated_mk_model = mc.generate_myokit_model()

            sim = mk.Simulation(generated_mk_model, mk_protocol)
            sim.set_tolerance(1e-9, 1e-9)
            sim.set_constant('markov_chain.E_Kr', Erev)
            sim.pre(5000)

            log = sim.run(tmax, log_times=times, log=['markov_chain.I_Kr'])
            gen_mk_IKr = np.array(log['markov_chain.I_Kr'])

            fig = plt.figure(figsize=(12, 9))
            fig.gca().plot(times[:-1], mk_IKr, label='original myokit simulation')
            fig.gca().plot(times[:-1], gen_mk_IKr, label='generated markov_builder simulation')

            fig.gca().legend()

            fig.savefig(os.path.join(comparison_plot_dir,
                                     f"{name}_myokit_comparison"))

            error = np.sqrt(np.mean((gen_mk_IKr - mk_IKr)**2))
            self.assertLess(error, 1e-2)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
