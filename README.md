# Markov-builder


## Description


## Getting Started

### Dependencies

It is advised to set up a virtual environment using `virtualenv venv` to avoid conflicts with other projects. This can be activated using `source venv/bin/activate`.

This library requires [PyGraphViz](https://pygraphviz.github.io/documentation/stable/install.html).

Once PyGraphViz is installed, the Python dependencies are provided in `requirements.txt` and may be installed by running `pip3 install -r requirements.txt`.

### Testing

In the base directory run `python3 -m unittest` to run all tests. By default, test output will be written to a folder inside `test` but this may be overwritten by exporting a `MARKOVBUILDER_TEST_OUTPUT` environment variable.
