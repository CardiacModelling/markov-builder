#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import math
import os
import pints
import symengine as se

def get_protocol_directory():
    return os.path.join(os.path.dirname( os.path.realpath(__file__)), "protocols")

def get_args(data_reqd=False, description=None):
    """Get command line arguments from using get_parser


    Params:

    data_reqd: is a flag which is set to True when a positional argument
    giving a path to some input data is needed

    description: is a string describing what the program should be used for. If
    this is none, get_parser uses a default description

    """
    # Check input arguments
    parser = get_parser(data_reqd, description=description)
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args


def get_parser(data_reqd=False, description=None):
    """Create an ArgumentParser object with common command line arguments already included

    Params:

    data_reqd: a flag which is set to True when a positional argument
    giving a path to some input data is needed

    description: is a string describing what the program should be used for. If
    this is none, get_parser uses a default description

    """

    if description is not None:
        description = 'Plot sensitivities of a Markov model'

    # Check input arguments
    parser = argparse.ArgumentParser(description=description)
    if data_reqd:
        parser.add_argument(
            "data_file_path",
            help="path to csv data for the model to be fit to",
            default=False)
    parser.add_argument(
        "-p",
        "--plot",
        action='store_true',
        help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100,
                        help="what DPI to use for figures")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="The directory to output figures and data to")
    return parser


def calculate_reversal_potential(temp=20):
    """
    Compute the Nernst potential of a potassium channel.

    Params:

    temp : temperature in degrees celcius, this defaults to 20

    """
    # E is the Nernst potential for potassium ions across the membrane
    # Gas constant R, temperature T, Faradays constat F
    R = 8314.55
    T = temp + 273.15
    F = 96485

    # Intracellular and extracellular concentrations of potassium.
    K_out = 4
    K_in = 130

    # valency of ions (1 in the case of K^+)
    z = 1

    # Nernst potential
    E = R * T / (z * F) * np.log(K_out / K_in)
    return E


def cov_ellipse(cov, offset=[0, 0], q=None,
                nsig=None, new_figure=True):
    """
    copied from stackoverflow
    Parameters
    ----------


    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * scipy.stats.norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')

    qs = np.sort(q)

    if not new_figure:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)

    for q in qs:
        r2 = scipy.stats.chi2.ppf(q, 2)
        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        rotation = np.arctan2(*vec[::-1, 0])

        # print("width, height, rotation = {}, {}, {}".format(width, height, math.degrees(rotation)))

        e = matplotlib.patches.Ellipse(offset,
                                       width[0],
                                       height[0],
                                       math.degrees(rotation),
                                       color=np.random.rand(3),
                                       fill=False,
                                       label="{}% confidence region".format(int(q * 100)))
        ax.add_patch(e)
        e.set_clip_box(ax.bbox)

        window_width = np.abs(width[0] * np.cos(rotation) * 1.5)
        window_height = np.abs(height[0] * np.sin(rotation) * 1.5)
        max_dim = max(window_width, window_height)

    if new_figure:
        ax.set_xlim(offset[0] - max_dim, offset[0] + max_dim)
        ax.set_ylim(offset[1] - max_dim, offset[1] + max_dim)
    return fig, ax


def extract_times(lst, time_ranges, step):
    """
    Take values from a list, lst which are indexes between the upper and lower
    bounds provided by time_ranges. Each element of time_ranges specifies an
    upper and lower bound.

    Returns a 2d numpy array containing all of the relevant data points
    """
    if time_ranges is None:
        return lst
    ret_lst = []
    for time_range in time_ranges:
        ret_lst.extend(lst[time_range[0]:time_range[1]:step].tolist())
    return np.array(ret_lst)


def remove_indices(lst, indices_to_remove):
    """Remove a list of indices from some list-like object

    Params:

    lst: a list-like object

    indices_to_remove: A list of pairs of indices (a,b) with a<b such that we
    remove indices strictly between a and b


    returns a new list

    """
    if indices_to_remove is None:
        return lst

    first_lst = lst[0:indices_to_remove[0][0]]

    lsts = []
    for i in range(1, len(indices_to_remove)):
        lsts.append(lst[indices_to_remove[i - 1][1]: indices_to_remove[i][0] + 1])

    lsts.append(lst[indices_to_remove[-1][1]:-1])

    lst = first_lst + [index for lst in lsts for index in lst]
    return lst


def detect_spikes(x, y):
    """
    Find the points where time-series 'jumps' suddenly. This is useful in order
    to find 'capacitive spikes' in protocols.

    Params:
    x : the independent variable (usually time)
    y : the dependent variable (usually voltage)

    """
    dx = np.diff(x)
    dy = np.diff(y)

    deriv = dy / dx
    spike_indices = np.argwhere(np.abs(deriv) > 10000)[:, 0]

    return x[spike_indices]


def beattie_sine_wave(t):
    """
    The sine-wave protocol from https://doi.org/10.1113/JP276068.

    Params:

    t: The time at which to compute the voltage


    Returns:
    V: the voltage at time t
    """

    # This shift is needed for simulated protocol to match the protocol
    # recorded in experiment, which is shifted by 0.1ms compared to the
    # original input protocol. Consequently, each step is held for 0.1ms
    # longer in this version of the protocol as compared to the input.
    shift = 0.1
    C = [54.0,
         26.0,
         10.0,
         0.007 / (2 * np.pi),
         0.037 / (2 * np.pi),
         0.19 / (2 * np.pi)]

    if t >= 250 + shift and t < 300 + shift:
        V = -120
    elif t >= 500 + shift and t < 1500 + shift:
        V = 40
    elif t >= 1500 + shift and t < 2000 + shift:
        V = -120
    elif t >= 3000 + shift and t < 6500 + shift:
        V = -30 + C[0] * (np.sin(2 * np.pi * C[3] * (t - 2500 - shift))) + C[1] * \
            (np.sin(2 * np.pi * C[4] * (t - 2500 - shift))) + \
            C[2] * (np.sin(2 * np.pi * C[5] * (t - 2500 - shift)))
    elif t >= 6500 + shift and t < 7000 + shift:
        V = -120
    else:
        V = -80
    return V


def get_protocol_from_csv(protocol_name : str, directory=None, holding_potential=-80):
    """Generate a function by interpolating
    time-series data.

    Params:
    Holding potential: the value to return for times outside of the
    range

    Returns:
    Returns a function float->float which returns the voltage (in mV)
    at any given time t (in ms)

    """

    if directory is None:
        directory = get_protocol_directory()

    protocol = pd.read_csv(os.path.join(directory, protocol_name+".csv"))

    times = 1000 * protocol["time"].values
    voltages = protocol["voltage"].values

    spikes = 1000 * detect_spikes(protocol["time"], protocol["voltage"])

    staircase_protocol = scipy.interpolate.interp1d(
        times, voltages, kind="linear")

    def staircase_protocol_safe(t): return staircase_protocol(
    t) if t < times[-1] and t > times[0] else holding_potential
    return staircase_protocol_safe


def draw_cov_ellipses(S1=None, sigma2=None, cov=None, plot_dir=None):
    """Plot confidence intervals using a sensitivity matrix or covariance matrix.

    In the case of a sensitivity matrix, i.i.d Guassian additive errors are
    assumed with variance sigma2. Exactly one of cov and S1 must not be None.
    In the case that S1 is provided, the confidence regions are calculated
    under the assumption that all other variables are fixed. However, if cov is
    provided, the confidence regions correspond a marginal distribution.

    Params:

    S1: A sensitivity matrix where S_i,j corresponds to the derivative
    of the (scalar) observation at the ith timepoint with respect to the jth
    parameter.

    sigma2: The variance of the Gaussian additive noise - only required if S1
    is provided

    cov: A covariance matrix

    plot_dir: The directory to store the plots in. When this defaults to None, the
    plots will be displayed using plt.show()

    """

    # TODO improve exception handling
    if S1 is not None:
        if cov is not None:
            Raise()
        else:
            n_params = S1.shape[1]
    else:
        if cov is None:
            Raise()
        if sigma2 is not None:
            Raise()
        else:
            n_params = cov.shape[0]

    for j in range(0, n_params - 1):
        for i in range(j + 1, n_params):
            if S1 is not None:
                if sigma2 is None:
                    raise
                sub_sens = S1[:, [i, j]]
                sub_cov = sigma2 * np.linalg.inv(np.dot(sub_sens.T, sub_sens))
            # Else use cov
            else:
                sub_cov = cov[parameters_to_view[:, None], np.array((i,j))]
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val = eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                # Parameters have been normalised to 1
                cov_ellipse(sub_cov, q=[0.5, 0.95], offset=[1, 1])
                plt.ylabel("parameter {}".format(i + 1))
                plt.xlabel("parameter {}".format(j + 1))
                plt.legend()
                if plot_dir is None:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            plot_dir,
                            "covariance_for_parameters_{}_{}".format(
                                j + 1,
                                i + 1)))
                plt.clf()
            else:
                print(
                    "COV_{},{} : negative eigenvalue: {}".format(
                        i, j, eigen_val))


def fit_model(funcs, data, starting_parameters, fix_parameters=None,
              max_iterations=None, method=pints.CMAES):
    """
    Fit a MarkovModel to some dataset using pints.

    Params:

    funcs: A MarkovModel

    data: The data set to fit to: a (1,n) numpy array
    consiting of observations corresponding to the times in funcs.times.

    starting_parameters: An initial guess for the optimal parameters

    fix_parameters: Which parameters (if any) should be ignored and set to fixed values

    max_iterations: An optional upper bound on the number of evaluations PINTS should perform

    method: Which optimisation method should be used

    returns: A pair containing the optimal parameters and the corresponding sum of square errors.

    """
    class Boundaries(pints.Boundaries):
        def __init__(self, parameters, fix_parameters=None):
            self.fix_parameters = fix_parameters
            self.parameters = parameters

        def check(self, parameters):
            '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
            '''
            # TODO Reimplement checks
            return True
            sim_params = np.copy(self.parameters)
            c = 0
            for i, parameter in enumerate(self.parameters):
                if i not in self.fix_parameters:
                    sim_params[i] = parameters[c]
                    c += 1
                if c == len(parameters):
                    break
            for i in range(0, 4):
                alpha = sim_params[2 * i]
                beta = sim_params[2 * i + 1]

                vals = [0, 0]
                vals[0] = alpha * np.exp(beta * -90 * 1E-3)
                vals[1] = alpha * np.exp(beta * 50 * 1E-3)

                for val in vals:
                    if val < 1E-7 or val > 1E3:
                        return False
            # Check maximal conductance
            if sim_params[8] > 0 and sim_params[8] < 2:
                return True
            else:
                return False

        def n_parameters(self):
            return 9 - \
                len(self.fix_parameters) if self.fix_parameters is not None else 9

    class PintsWrapper(pints.ForwardModelS1):
        def __init__(self, funcs, parameters, fix_parameters=None):
            self.funcs = funcs
            self.parameters = parameters
            self.fix_parameters = fix_parameters
            if fix_parameters is not None:
                self.free_parameters = [i for i in range(
                    0, len(starting_parameters)) if i not in fix_parameters]
            else:
                self.free_parameters = range(0, len(starting_parameters))

        def n_parameters(self):
            if self.fix_parameters is not None:
                return len(self.parameters) - len(self.fix_parameters)
            else:
                return len(self.parameters)

        def simulate(self, parameters, times):
            self.funcs.times = times
            if self.fix_parameters is None:
                return self.funcs.SimulateForwardModel(parameters, times)
            else:
                sim_params = np.copy(self.parameters)
                c = 0
                for i, parameter in enumerate(self.parameters):
                    if i not in self.fix_parameters:
                        sim_params[i] = parameters[c]
                        c += 1
                    if c == len(parameters):
                        break
                return self.funcs.SimulateForwardModel(sim_params, times)

        def simulateS1(self, parameters, times):
            if fix_parameters is None:
                return self.funcs.SimulateForwardModelSensitivities(parameters)
            else:
                sim_params = np.copy(self.parameters)
                c = 0
                for i, parameter in enumerate(self.parameters):
                    if i not in fix_parameters:
                        sim_params[i] = parameters[c]
                        c += 1
                    if c == len(parameters):
                        break
                current, sens = self.funcs.SimulateForwardModelSensitivities(
                    sim_params, times)
                print(sim_params)
                sens = sens[:, self.free_parameters]
                return current, sens

    model = PintsWrapper(funcs, starting_parameters,
                         fix_parameters=fix_parameters)
    problem = pints.SingleOutputProblem(model, funcs.times, data)
    error = pints.SumOfSquaresError(problem)
    boundaries = Boundaries(starting_parameters, fix_parameters)

    print("data size is {}".format(data.shape))

    if fix_parameters is not None:
        params_not_fixed = [starting_parameters[i] for i in range(
            len(starting_parameters)) if i not in fix_parameters]
    else:
        params_not_fixed = starting_parameters
    print(params_not_fixed)
    controller = pints.OptimisationController(
        error, params_not_fixed, boundaries=boundaries, method=method)
    if max_iterations is not None:
        print("Setting max iterations = {}".format(max_iterations))
        controller.set_max_iterations(max_iterations)

    found_parameters, found_value = controller.run()
    return found_parameters, found_value


def get_protocol(protocol_name : str):
    """Returns a function describing the voltage trace.

    params:

    protocol_name: A string used to select the protocol

    returns: A function, v(t), which returns the voltage (mV)
    at a time t (ms).

    """
    v = None
    if protocol_name == "sine-wave":
        v = beattie_sine_wave
    else:
        # Check the protocol folders for a protocol with the same name
        protocol_dir = get_protocol_directory()
        possible_protocol_path = os.path.join(protocol_dir, protocol_name+".csv")
        if os.path.exists(possible_protocol_path):
            try:
                v = get_protocol_from_csv(protocol_name)
            except:
                # TODO
                raise
        else:
            # Protocol not found
            raise Exception("Protocol not found at " + possible_protocol_path)
    return v
