#
# markov_builder setuptools script
#
import os

from setuptools import find_packages, setup


# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number
with open(os.path.join('markov_builder', 'version.txt'), 'r') as f:
    version = f.read()

# Go!
setup(
    # Module name (lowercase)
    name='markov_builder',

    version=version,
    description='markov builder for cardiac modelling',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Joseph Shuttleworth, Dominic Whittaker, Michael Clerx, Maurice Hendrix, Gary Mirams',
    author_email='joseph.shuttleworth@nottingham.ac.uk',
    maintainer='Maurice Hendrix',
    maintainer_email='joseph.shuttleworth@nottingham.ac.uk',
    url='https://github.com/CardiacModelling/markov-builder',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # Packages to include
    packages=find_packages(
        include=('markov_builder', 'markov_builder.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.6',

    # List of dependencies
    install_requires=[
        #'pints>=0.3',
        #'scipy>=1.7',
        #'numpy>=1.21',
        #'matplotlib>=3.4',
        #'pandas>=1.3',
        #'networkx>=2.6',
        #'plotly>=5.3',
        #'symengine>=0.8',
        #'sympy>=1.8',
        #'pygraphviz==1.7',
    ],
    extras_require={
        'test': [
            #'pytest-cov>=2.10',     # For coverage checking
            #'pytest>=4.6',          # For unit tests
            #'flake8>=3',            # For code style checking
            #'isort',
            #'mock>=3.0.5',         # For mocking command line args etc.
            #'codecov>=2.1.3',
        ],
    },
)
