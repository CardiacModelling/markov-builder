#!/usr/bin/env python3

# Simulate data from the Beattie model and M10 model using a Gillespie
# algorithm output plots into examples/example_output or
# MARKOVBUILDER_EXAMPLE_OUTPUT if it exists

import matplotlib.pyplot as plt
import os
import sys
import logging
import numpy as np
import pandas as pd
from markov_builder import construct_four_state_chain, construct_M10_chain


def main():
    # First define functions which output the values of each transition rate
    # for a given voltage (as dictionaries)
    def beattie_get_rates(voltage: float, parameters: list):
        # Now get the waiting times and embedded MC
        rates = [parameters[2 * i] + np.exp(parameters[2 * i + 1] * voltage)
                    for i in range(int((len(parameters) - 1) / 2))]

        rate_vals = {"k1": rates[0],
                        "k2": rates[1],
                        "k3": rates[2],
                        "k4": rates[3]
                        }
        return rate_vals

    def M10_get_rates(voltage: float, params: list):
        # Now get the waiting times and embedded MC
        rates = [params[2 * i] + np.exp(params[2 * i + 1] * voltage) for i in range(int((len(params)) / 2))]
        rate_vals = dict(zip(('a1', 'b1', 'bh', 'ah', 'a2', 'b2'), rates))
        return rate_vals

    beattie_params = np.array([2.07E-3, 7.17E-2, 3.44E-5, -6.18E-2, 20, 2.58E-2, 2,
                               2.51E-2, 3.33E-2])

    M10_params = (8.5e-03, 8.32e-02, 1.3e-02, 1.0e-07, 2.7e-01, 1.6e-02,
                  7.7e-02, 2.2e-02, 1.5e-01, 2.4e-02, 5.6e-04,
                  4.1e-02)

    # Perform the simulations
    mc = construct_four_state_chain()
    protocol = ((-80, 100), (20, 200))
    SimulateStepProtocol(mc, beattie_get_rates, protocol, beattie_params, name="Beattie")

    # M10-IKr model
    mc = construct_M10_chain()
    SimulateStepProtocol(mc, M10_get_rates, protocol, M10_params, name="M10")

def SimulateStepProtocol(mc, rates_func, protocol, params, name: str = ""):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    no_trajectories = 100
    dist = None
    data = [pd.DataFrame(columns=("time", *mc.graph.nodes))]
    last_time = 0
    eqm_data = []
    for voltage, time_to in protocol:
        data.append(mc.sample_trajectories(no_trajectories, rates_func(
            voltage, params), (last_time, time_to), starting_distribution=dist))
        dist = data[-1].values[-1, 1:]
        _, A = mc.eval_transition_matrix(rates_func(voltage, params))
        # compute steady states
        labels, ss = mc.get_equilibrium_distribution(rates_func(voltage, params))
        ss = ss * no_trajectories
        eqm_data = eqm_data + [[last_time, *ss]] + [[time_to, *ss]]
        last_time = time_to

    eqm_data = pd.DataFrame(eqm_data, columns=['time'] +
                            [l + ' eqm distribution' for l in labels]).set_index("time")
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
