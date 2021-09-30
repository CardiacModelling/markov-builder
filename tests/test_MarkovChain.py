#!/usr/bin/env python3

import logging
import os
import sys
import unittest

import networkx as nx
import sympy as sp

import markov_builder.example_models as example_models


class TestMarkovChain(unittest.TestCase):

    def setUp(self):
        """Run by unittest before the tests in this class are performed.

        Create an output directory (if it doesn't already exist). An
        alternative output path can be used by setting the
        MARKOVBUILDER_TEST_OUTPUT environment variable

        """
        test_output_dir = os.environ.get('MARKOVBUILDER_TEST_OUTPUT', os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.__class__.__name__))
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_construct_chain(self):
        """Construct various examples of Markov models.
        Output dot files of these graphs in the test output directory.

        Check that our transition rate matrix for the Beattie model is correct
        by comparing it against a reference solution.

        TODO: Add a reference solution for the m10 model

        """

        # Construct Beattie model
        logging.info("Constructing four-state Beattie model")

        mc = example_models.construct_four_state_chain()

        # Output DOT file
        nx.drawing.nx_agraph.write_dot(mc.graph, "Beattie_dotfile.dot")

        # Draw graph using pyvis
        mc.draw_graph(os.path.join(self.output_dir, "BeattieModel.html"))
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
        logging.info("Constructing six-state M10 model")

        m10 = example_models.construct_M10_chain()

        # Save DOTfile
        nx.drawing.nx_agraph.write_dot(m10.graph, "M10_dotfile.dot")

        # Save html visualisation using pyvis
        m10.draw_graph(os.path.join(self.output_dir, "M10.html"))

        # Construct Mazhari model
        logging.info("Constructing five-state Mazhari model")

        mazhari = example_models.construct_mazhari_chain()

        # Save DOTfile
        nx.drawing.nx_agraph.write_dot(mazhari.graph, "Mazhari_dotfile.dot")
        
        # Save html visualisation using pyvis
        mazhari.draw_graph(os.path.join(self.output_dir, "Mazhari.html"))

    def test_construct_open_trapping_model(self):
        """

        Construct a model then run the add_open_trapping function to mirror the
        model and add drug trapping to the open channel.

        """

        models = (example_models.construct_four_state_chain(), example_models.construct_M10_chain(),
                  example_models.construct_non_reversible_chain())

        for mc in models:
            mc.add_open_trapping(prefix="d_", new_rates=True)

            # Save dotfile
            nx.drawing.nx_agraph.write_dot(mc.graph, os.path.join(self.output_dir, "open_trapping.dot"))

            mc.draw_graph(os.path.join(self.output_dir, "%s_open_trapping.html" % mc.name))
            logging.debug(mc.graph)

    def test_assert_reversibility_using_cycles(self):
        """Test that MarkovChain().is_reversible correctly identifies if markov
        chains are reversible or not.

        MarkovChain().is_reversible checks reversibility using the method
        presented in Colquhoun et al. (2014)
        https://doi.org/10.1529/biophysj.103.038679

        TODO add more test cases

        """

        models = [example_models.construct_four_state_chain(), example_models.construct_M10_chain(), 
                  example_models.construct_mazhari_chain()]

        m = 0
        for mc in models:
            m += 1
            logging.info("Checking reversibility")
            logging.info("Model = " + str(m))
            assert(mc.is_reversible())
            logging.info("Checking reversibility with open trapping")
            mc.add_open_trapping(new_rates=True)
            assert(mc.is_reversible())

        # Test is_reversible on a model we know is not reversible. This example
        # is a simple three state model with 6 independent transition rates

        logging.info("Checking reversibility of non-reversible chain")
        mc = example_models.construct_non_reversible_chain()
        logging.debug("graph is %s", mc.graph)

        assert(not mc.is_reversible())
        logging.info("Checking reversibility of non-reversible chain")
        mc.add_open_trapping()
        logging.debug("graph is %s", mc.graph)
        assert(not mc.is_reversible())

    def test_equate_rates(self):
        """
        Test that the MarkovChain.substitute_rates function performs the expected substitution

        TODO: add more test cases
        """
        mc = example_models.construct_four_state_chain()
        rates_dict = {'k1': 'k2*k4'}
        mc.substitute_rates(rates_dict)
        mc.draw_graph(os.path.join(self.output_dir, '%s_rates_substitution.html' % mc.name), show_rates=True)

        label_found = False
        for u, v, d in mc.graph.edges(data=True):
            if 'label' in d:
                label_found = True
                if d['label'] == 'k1':
                    self.assertEqual(rates_dict['k1'], d['rate'])
        self.assertTrue(label_found)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
