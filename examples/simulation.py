#!/usr/bin/env python3

# Simulate data from the Beattie model and M10 model using a Gillespie
# algorithm output plots into examples/example_output or
# MARKOVBUILDER_EXAMPLE_OUTPUT if it exists

import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from markov_builder import example_models


def main():
    # First define functions which output the values of each transition rate
    # for a given voltage (as dictionaries)
    # Perform the simulations
    mc = example_models.construct_four_state_chain()
    protocol = ((-80, 100), (20, 200))
    SimulateStepProtocol(mc, protocol, name="Beattie")


def SimulateStepProtocol(mc, protocol, name: str = ""):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    no_trajectories = 100
    dist = None
    data = [pd.DataFrame(columns=("time", *mc.graph.nodes))]
    last_time = 0
    eqm_data = []
    param_dict = mc.default_values

    for voltage, time_to in protocol:
        param_dict['V'] = voltage

        data.append(mc.sample_trajectories(no_trajectories, (last_time, time_to),
                                           starting_distribution=dist, param_dict=param_dict))
        dist = data[-1].values[-1, 1:]
        _, A = mc.eval_transition_matrix(param_dict)
        # compute steady states
        labels, ss = mc.get_equilibrium_distribution(param_dict=param_dict)
        ss = ss * no_trajectories
        eqm_data = eqm_data + [[last_time, *ss]] + [[time_to, *ss]]
        last_time = time_to

    eqm_data = pd.DataFrame(eqm_data, columns=['time'] + [lb + ' eqm distribution' for lb in labels]).set_index("time")

    data = pd.concat(data).set_index("time").sort_index()

    data.plot(ax=ax1)
    eqm_data.plot(style="--", ax=ax1)

    ax2 = fig.add_subplot(212)

    # Need each voltage twice - at the beginning and end of each step
    voltages = [[v, v] for v, _ in protocol]
    voltages = [v for voltage in voltages for v in voltage]
    times = [0]
    for _, time_to in protocol:
        times = times + [time_to] * 2
    times = times[0:-1]
    ax2.plot(times, voltages)
    fpath = os.path.join(output_dir, "SimulateStepProtocol_{}.pdf".format(name))
    plt.savefig(fpath)
    logging.info("wrote to %s" % fpath)


if __name__ == "__main__":
    global output_dir
    output_dir = os.environ.get('MARKOVBUILDER_EXAMPLE_OUTPUT', os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "example_output"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        output_dir = output_dir
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info("outputting to " + output_dir)
    main()
