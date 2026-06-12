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
import pytest
import sympy as sp

import markov_builder
import markov_builder.models.thirty_models


matplotlib.use('pdf')

models = [getattr(markov_builder.models.thirty_models, f"model_{i:02}") for i in range(31)]
disconnected_models = [models[i] for i in [3, 9, 10, 19, 20, 26, 27]]

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


@pytest.mark.parametrize("model", set(models) - set(disconnected_models))
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
    sim.pre(10000)

    log = sim.run(tmax, log_times=times, log=['ikr.IKr'])

    mk_IKr = np.array(log['ikr.IKr'])

    remove = None
    if model not in disconnected_models + [models[0], models[1], models[8]]:
        # Find removed state
        for state_var in mk_model["ikr"]:
            str_state_var = str(state_var).split(".")[1]
            print(state_var)
            if (not state_var.is_state()) and (str_state_var in mc.graph):
                remove = str_state_var
                break
        if remove is None:
            assert remove, "Couldn't find redundant state to remove"

    generated_mk_model = mc.generate_myokit_model(eliminate_state=remove)

    sim = mk.Simulation(generated_mk_model, mk_protocol)
    sim.set_tolerance(1e-9, 1e-9)
    sim.set_constant('markov_chain.E_Kr', Erev)
    sim.pre(10000)

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

    test_threshold = 0.02 if model in disconnected_models else 1e-5

    assert error < test_threshold, f"Excessive error in model {index}: {name} {error} >= {test_threshold}"


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
