# Markov-builder


## Description


## Getting Started

### Dependencies

It is advised to set up a virtual environment using `virtualenv --python=python3 venv` to avoid conflicts with other projects. This can be activated using `source venv/bin/activate`.

This library requires [GraphViz](https://graphviz.org/).

Once GraphViz is installed, the Python dependencies are provided in `requirements.txt` and may be installed by running `pip install -r requirements.txt`.

To install the _markov_builder_ package itself, run `pip install -e .`

### Testing

In the base directory run `python -m unittest` to run all tests. By default, test output will be written to a folder inside `test` but this may be overwritten by exporting a `MARKOVBUILDER_TEST_OUTPUT` environment variable.

To see detailed output from the tests, simply execute them in the `tests/` folder, e.g. `cd tests/` followed by `python test_MarkovChain.py`.
