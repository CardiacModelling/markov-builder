#!/usr/bin/env python3

import logging
import os
import unittest

import myokit
import networkx as nx

from markov_builder.models.thirty_models import (
    model_00,
    model_01,
    model_02,
    model_03,
    model_04,
    model_05,
)


class TestThirtyModels(unittest.TestCase):

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

        self.models = [model_00, model_01, model_02, model_03, model_04, model_05]

    def test_generate_myokit(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"intiating {name}")
            mc = model()
            myokit_model = mc.generate_myokit_model()
            myokit.save(os.path.join(self.output_dir, f"{model.__name__}.mmt"), myokit_model)
        return

    def test_visualise_graphs(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"intiating {name}")
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
            logging.debug(f"intiating {name}")

            mc = model()
            self.assertTrue(mc.is_connected())

    def test_reversible(self):
        for model in self.models:
            name = model.__name__
            logging.debug(f"intiating {name}")

            mc = model()
            self.assertTrue(mc.is_reversible())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
