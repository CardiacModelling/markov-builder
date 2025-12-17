#!/usr/bin/env python3

import itertools
import pytest
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

import markov_builder
import markov_builder.models.thirty_models

from markov_builder.models.thirty_models import (model_00, model_01, model_02,
                                                 model_03, model_04, model_05,
                                                 model_06, model_07, model_08,
                                                 model_09, model_10, model_11,
                                                 model_12, model_13, model_14,
                                                 model_15, model_16, model_17,
                                                 model_18, model_19, model_20,
                                                 model_21, model_22, model_23,
                                                 model_24, model_25, model_26,
                                                 model_27, model_28, model_29,
                                                 model_30)


matplotlib.use('pdf')

models = [getattr(markov_builder.models.thirty_models, f"model_{i:02}") for i in range(31)]


disconnected_models = [model_03, model_09, model_10, model_19,
                       model_20, model_26, model_27]

output_dir = os.environ.get('MARKOVBUILDER_TEST_OUTPUT', 'test_output')
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logging.info("outputting to " + output_dir)


@pytest.mark.parametrize("model", models)
def test_generate_myokit(model):
    name = model.__name__
    logging.debug(f"initiating {name}")
    mc = model()
    myokit_model = mc.generate_myokit_model()
    myokit.save(os.path.join(output_dir, f"{model.__name__}.mmt"), myokit_model)

@pytest.mark.parametrize("model", models)
def test_visualise_graphs(model):
    name = model.__name__
    logging.debug(f"initiating {name}")
    mc = model()

    mc.draw_graph(os.path.join(output_dir, f"{name}_graph.html"),
                    show_parameters=False)
    mc.draw_graph(os.path.join(output_dir, f"{name}_graph_with_parameters.html"),
                    show_parameters=True)

    nx.drawing.nx_agraph.write_dot(mc.graph, os.path.join(output_dir,
                                                            "%s_dotfile.dot" % name))

@pytest.mark.parametrize("model", models)
def test_connected(model):
    name = model.__name__
    logging.debug(f"initiating {name}")

    mc = model()
    assert mc.is_connected() ^ (model in disconnected_models), f"model {model} is not connected"

@pytest.mark.parametrize("model", models)
def test_reversible(model):
    name = model.__name__
    logging.debug(f"initiating {name}")

    if model in disconnected_models:
        return

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
                logging.error("%s: rates moving forwards around the cycle are: %s", name, forward_rate_list)
                logging.error("%s: rates moving backwards around the cycle are: %s", name, backward_rate_list)
                logging.error(f"States are {cycle}")

    assert mc.is_reversible(), "MC is not reversible"


@pytest.mark.parametrize("model", models)
def test_myokit_simulation_output(model):
    mmt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'models_myokit')
    Erev = -88

    # Extract number from model name i.e. model_02 -> 02
    index = int(str(model.__name__)[-2:])

    comparison_plot_dir = os.path.join(output_dir, 'comparison_plots')

    if not os.path.exists(comparison_plot_dir):
        os.makedirs(comparison_plot_dir)

    name = model.__name__

    mc = model()
    mk_protocol_filename = os.path.join(mmt_dir,
                                        'simplified-staircase.mmt')

    mk_protocol = mk.load_protocol(mk_protocol_filename)

    mk_model = mk.load_model(os.path.join(mmt_dir,
                                            f"model-{index}.mmt"))

    # logging.debug(f"Loaded model-{i}.mmt")

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
    plt.close(fig)

    error = np.sqrt(np.mean((gen_mk_IKr - mk_IKr)**2))
    assert error < 0.02, f"Excessive error in model {index}: {name}"


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
