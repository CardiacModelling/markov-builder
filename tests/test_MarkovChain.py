#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import myokit
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

        self.models = [example_models.construct_four_state_chain(), example_models.construct_M10_chain(),
                       example_models.construct_non_reversible_chain(), example_models.construct_mazhari_chain(),
                       example_models.construct_wang_chain()]

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

        system = mc.eliminate_state_from_transition_matrix(['C', 's_O', 's_I'])

        pen_and_paper_A = sp.Matrix([['-k_1 - k_3 - k_4', 'k_2 - k_4', '-k_4'],
                                     ['k_1', '-k_2 - k_3', 'k_4'],
                                     ['-k_1', 'k_3 - k_1', '-k_2 - k_4 - k_1']])
        pen_and_paper_B = sp.Matrix(['k_4', 0, 'k_1'])

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

        # Construct Wang model
        logging.info("Constructing five-state Wang model")

        wang = example_models.construct_wang_chain()

        # Save DOTfile
        nx.drawing.nx_agraph.write_dot(wang.graph, "Wang_dotfile.dot")

        # Save html visualisation using pyvis
        wang.draw_graph(os.path.join(self.output_dir, "Wang.html"))

    def test_parameterise_rates(self):
        """
        Test the MarkovChain.parameterise_rates function.
        """

        mc = example_models.construct_four_state_chain()

        # Expressions to be used for the rates. The variable V (membrane
        # voltage) is shared across expressions and so it should only appear
        # once in the parameter list.

        # Output system of equations
        logging.debug("ODE system is %s", str(mc.get_transition_matrix(use_parameters=True)))

        # Output reduced system of equations
        logging.debug("Reduced ODE system is :%s",
                      str(mc.eliminate_state_from_transition_matrix(list(mc.graph.nodes)[:-2],
                                                                    use_parameters=True)))

        # Output list of parameters
        param_list = mc.get_parameter_list()
        logging.debug("parameters are %s", mc.get_parameter_list())

        self.assertEqual(param_list.count('V'), 1)

        # Generate myokit code
        myokit_model = mc.generate_myokit_model()
        myokit.save(os.path.join(self.output_dir, 'beattie_model.mmt'), myokit_model)

        myokit_model = mc.generate_myokit_model(eliminate_state='IC')
        myokit.save(os.path.join(self.output_dir, 'beattie_model_reduced.mmt'), myokit_model)

    def test_construct_open_trapping_model(self):
        """

        Construct a model then run the add_open_trapping function to mirror the
        model and add drug trapping to the open channel.

        """

        for mc in self.models:
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

        # Test function on models that we know are reversible
        reversible_models = [example_models.construct_four_state_chain(),
                             example_models.construct_M10_chain(),
                             example_models.construct_mazhari_chain()]

        for mc in reversible_models:
            logging.info("Checking reversibility")
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
        rates_dict = {'k_1': 'k_2*k_4'}
        mc.substitute_rates(rates_dict)
        mc.draw_graph(os.path.join(self.output_dir, '%s_rates_substitution.html' % mc.name), show_rates=True)
        label_found = False
        for _, _, d in mc.graph.edges(data=True):
            if 'label' in d:
                label_found = True
                if d['label'] == 'k1':
                    self.assertEqual(rates_dict['k1'], d['rate'])
        self.assertTrue(label_found)

        mc = example_models.construct_four_state_chain()
        mc.substitute_rates(rates_dict)
        mc.add_open_trapping(new_rates=True)
        mc.draw_graph(os.path.join(self.output_dir,
                                   '%s_open_trapping_rates_substitution.html' %
                                   mc.name), show_rates=True)
        transition_rates = [d['rate'] for _, _, d in mc.graph.edges(data=True)]

        # Check that new mirrored rates have been handled correctly
        self.assertIn('d_k_2*d_k_4', transition_rates)
        self.assertIn('d_k_2', mc.rates)
        self.assertIn('d_k_4', mc.rates)

        # Check reversibility still holds for good measure
        self.assertTrue(mc.is_reversible())

    def test_latex_printing(self):
        """ Test that we can generate LaTeX expressions for the four state model

        TODO: Add more cases
        """
        mc = example_models.construct_four_state_chain()
        logging.debug(mc.as_latex())
        logging.debug(mc.as_latex(state_to_remove='s_O'))
        logging.debug(mc.as_latex(include_auxiliary_expression=True))
        logging.debug(mc.as_latex('s_O', True))

    def test_sample_trajectories(self):
        """Simulate the 4-state Beattie Model using the Gillespie method

        TODO add more models
        """
        mc = example_models.construct_four_state_chain()
        df = mc.sample_trajectories(1000, (0, 100), {'V': 0})
        df = df.set_index('time')
        print(df)
        df.plot()
        plt.savefig(os.path.join(self.output_dir, 'beattie_model_sample_trajectories'))
        logging.debug


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
