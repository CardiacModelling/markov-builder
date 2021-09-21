#!/usr/bin/env python3

import sys
import os
import logging
import unittest
import MarkovModels
import sympy as sp
import networkx as nx

from MarkovModels import MarkovChain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def construct_M10_chain():
    mc = MarkovChain()

    mc.add_states(('IC1','IC2','IO','C1','C2'))
    mc.add_state('O', open=True)
    rates = (('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'), ('IO', 'O', 'ah', 'bh'), ('O', 'C1', 'b2', 'a2'), ('C1', 'C2', 'b1', 'a1'), ('C2', 'IC2', 'bh', 'ah'), ('C1', 'IC1', 'bh', 'ah'))

    for r in rates:
        mc.add_both_transitions(*r)
    return mc

def construct_non_reversible_chain():
    mc = MarkovChain()

    mc.add_states(('A', 'D'))
    mc.add_state('B', open=True)
    rates = (('B', 'A', 'k1', 'k2'), ('A', 'D', 'k3', 'k4'), ('B', 'D', 'k5', 'k6'))

    for r in rates:
        mc.add_both_transitions(*r)
    return mc

def construct_four_state_chain():
    mc = MarkovChain()
    rates = ['k{}'.format(i) for i in [1,2,3,4]]
    mc.add_rates(rates)
    states = [('O', True), ('C', False), ('I', False), ('IC', False)]
    mc.add_states(states)
    mc.add_state('O', open=True)

    rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'), ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

    for r in rates:
        mc.add_both_transitions(*r)
    return mc

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        test_output_dir = os.environ.get('MARKOVMODELS_TEST_OUTPUT', os.path.join(os.path.dirname(os.path.realpath(__file__)), self.__class__.__name__))
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_construct_open_trapping_model(self):
        mc = construct_four_state_chain()
        mc.add_open_trapping(prefix="d_", new_rates=True)

        labels = ('O', 'C', 'I', 'd_O', 'd_C', 'd_I', 'd_O', 'd_IC')
        eqns = mc.eliminate_state_from_transition_matrix(labels)
        logging.debug("System of equations for open trapping model is {}".format(eqns))

        nx.drawing.nx_agraph.write_dot(mc.graph, os.path.join(self.output_dir, "open_trapping.dot"))
        logging.debug(mc.graph)



    def test_construct_chain(self):
        logging.info("Constructing four-state Beattie model")

        mc = construct_four_state_chain()
        pos=nx.spring_layout(mc.graph)

        # Output DOT file
        nx.drawing.nx_agraph.write_dot(mc.graph, "Beattie_dotfile.dot")

        logging.debug(mc.graph)

        labels, Q = mc.get_transition_matrix()
        logging.debug("Q^T matrix is {}, labels are {}".format(Q.T, labels))

        system = mc.eliminate_state_from_transition_matrix(['C', 'O', 'I'])

        pen_and_paper_A = sp.Matrix([['-k1 - k3 - k4', 'k2 - k4', '-k4'],
                    ['k1', '-k2 - k3', 'k4'], ['-k1', 'k3 - k1', '-k2 - k4 - k1']])

        pen_and_paper_B = sp.Matrix(['k4', 0, 'k1'])

        self.assertEqual(pen_and_paper_A, system[0])
        self.assertEqual(pen_and_paper_B, system[1])

        # Construct M10 model
        m10 = construct_M10_chain()
        nx.drawing.nx_agraph.write_dot(m10.graph, "m10_dotfile.dot")

    def test_assert_reversibility_using_cycles(self):
        models = [construct_four_state_chain(), construct_M10_chain()]

        for mc in models:
            logging.info("Checking reversibility")
            assert(mc.is_reversible())
            logging.info("Checking reversibility with open trapping")
            mc.add_open_trapping(new_rates=True)
            assert(mc.is_reversible())

        logging.info("Checking reversibility of non-reversible chain")
        mc = construct_non_reversible_chain()
        logging.debug("graph is %s", mc.graph)

        assert(not mc.is_reversible())
        logging.info("Checking reversibility of non-reversible chain")
        mc.add_open_trapping()
        logging.debug("graph is %s", mc.graph)
        assert(not mc.is_reversible())

    def test_SimulateStepProtocol(self):
        # First test the Beattie model
        def beattie_get_rates(voltage : float, parameters : list):
            # Now get the waiting times and embedded MC
            rates=[parameters[2*i]+np.exp(parameters[2*i+1]*voltage) for i in range(int((len(parameters)-1)/2))]

            rate_vals = {"k1" : rates[0],
                    "k2" : rates[1],
                    "k3" : rates[2],
                    "k4" : rates[3]
            }
            return rate_vals

        mc = construct_four_state_chain()
        protocol = ((-80, 100), (20, 200))
        self.SimulateStepProtocol(mc, beattie_get_rates, protocol, name="Beattie")

        # M10-IKr model
        mc = construct_M10_chain()
        def M10_get_rates(voltage : float, params : list):
            params = (8.53183002138620944e-03, 8.31760044455376601e-02, 1.26287052202195688e-02, 1.03628499834739776e-07, 2.70276339808042609e-01,1.58000446046794897e-02, 7.66699486356391818e-02, 2.24575000694940963e-02, 1.49033896782688496e-01, 2.43156986537036227e-02, 5.58072076984100361e-04, 4.06619125485430874e-02)
            # Now get the waiting times and embedded MC
            rates=[params[2*i]+np.exp(params[2*i+1]*voltage) for i in range(int((len(params))/2))]
            rate_vals = dict(zip(('a1', 'b1', 'bh', 'ah', 'a2', 'b2'), rates))
            return rate_vals

        self.SimulateStepProtocol(mc, M10_get_rates, protocol, name="M10")


    def SimulateStepProtocol(self, mc, rates_func, protocol, name : str =""):
        params = np.array([2.07E-3, 7.17E-2, 3.44E-5, -6.18E-2, 20, 2.58E-2, 2,
                           2.51E-2, 3.33E-2])
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(211)
        no_trajectories = 10
        dist=None
        data = [pd.DataFrame(columns=("time", *mc.graph.nodes))]
        last_time=0
        eqm_data = []
        for voltage, time_to in protocol:
            data.append(mc.sample_trajectories(no_trajectories, rates_func(voltage, params), (last_time, time_to), starting_distribution=dist))
            dist = data[-1].values[-1,1:]
            _,A = mc.eval_transition_matrix(rates_func(voltage, params))
            # compute steady states
            labels, ss = mc.get_equilibrium_distribution(rates_func(voltage, params))
            ss = ss*no_trajectories
            eqm_data=eqm_data + [[last_time, *ss]] + [[time_to, *ss]]
            last_time = time_to

        eqm_data = pd.DataFrame(eqm_data, columns = ['time'] + [l + ' eqm distribution' for l in labels]).set_index("time")
        data=pd.concat(data).set_index("time").sort_index()

        data.plot(ax=ax1)
        eqm_data.plot(style="--", ax=ax1)

        ax2= fig.add_subplot(212)

        # Need each voltage twice - at the beginning and end of each step
        voltages = [[v, v] for v,_ in protocol]
        voltages = [v for voltage in voltages for v in voltage]
        times=[0]
        for _, time_to in protocol:
            times=times+[time_to]*2
        times=times[0:-1]
        ax2.plot(times,voltages)
        plt.savefig(os.path.join(self.output_dir, "SimulateStepProtocol_{}.pdf".format(name)))

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
