# Markov-builder


## Description
`markov_builder` is a Python package for constructing Markov models of ion-channel kinetics. These models may be thought of as chemical reaction networks where the transition rate matrix depends on the cell's transmembrane potential (and sometimes the concentration of a drug). This package uses 'networkx' to allow the user to specify the graph associated with the model (model topology). Many models of ion channel kinetics can be expressed in this way. For example, Markov models equivalent to Hodgkin-Huxley style conductance models are straightforward to generate. The transition rates may then be parameterised with any choice of equation, and depend on any number of 'shared variables' such as transmembrane potential, drug concentration, temperature, etc... 

Given the graph for a model and a parameterisation, you may then:
1. Output latex describing the model
2. Visualise the model using pyvis
3. Output the system of equations as `sympy` expressions for use in other code
4. Model drug trapping by composing models together
5. Simulate a small number of channels using the Gillespie algorithm
6. Check the model satisfies properties such as microscopic reversibility and connectedness 


## Getting Started

### Dependencies
This package requires Python with version >= 3.6 with `numpy`. `pip` is required for installation.

### Installation

It is recommended to install `markov_builder` in a virtual environment to avoid dependency conflicts. To do this, navigate to this repository and run:

```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade venv
python3 -m venv .venv
activate .venv/bin/activate
```
Then install `markov_builder`, along with its dependencies.
```
python3 -m pip install .
```

### Testing
If you wish to run the tests, you must first install the test dependencies.
python3 -m pip install .[test]

Then run the tests.
```
pip3 -m pytest
```
By default, test output will be written to a folder inside `tests` but this may be overwritten by exporting a `MARKOVBUILDER_TEST_OUTPUT` environment variable.

To see detailed output from the tests, simply execute them e.g. `python -m tests.test_MarkovChain.py`.
