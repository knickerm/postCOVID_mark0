"""
Generic implementation of a class that works as a generic interface for the
models to be called and run using Python
"""

__author__ = "Karl Naumann-Woleske, Max Sina Knicker"
__credits__ = ["Max Sina Knicker", "Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Max Sina Knicker"

import os

import numpy as np
import pandas as pd
import yaml
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.cm import get_cmap

# General Font settings
x = r'\usepackage[bitstream-charter, greekfamily=default]{mathdesign}'
rc('text.latex', preamble=x)
rc('text', usetex=True)
rc('font', **{'family': 'serif'})

# Font sizes
base = 12
rc('axes', titlesize=base - 2)
rc('legend', fontsize=base - 2)
rc('axes', labelsize=base - 2)
rc('xtick', labelsize=base - 3)
rc('ytick', labelsize=base - 3)

# Axis styles
cycles = cycler('linestyle', ['-', '--', ':', '-.'])
cmap = get_cmap('gray')
cycles += cycler('color', cmap(list(np.linspace(0.1, 0.9, 4))))
rc('axes', prop_cycle=cycles)


class AgentBasedModel(object):
    """ Generic class representing an Agent-based Model. This class ensures that
    a minimum set of universal functions is available for each ABM class, to
    ensure correct functioning of the model in the analysis

    Attributes
    ----------
    parameters : dict
        The dictionary of parameters that define the model as a whole
    hyper_parameters : dict
        The parameters defining the simulations, e.g. number of firms, seed,
        runtime
    output : pd.DataFrame, default: None
        A dataframe of the time-series generated by the simulation
    """

    def __init__(self, parameters: dict = None, hyper_parameters: dict = None):
        """ Initialisation of an Agent-based Model on the basis of

        Parameters
        ----------
        parameters : dict, default: None
            Parameters that define the model
        hyper_parameters : dict, default: None
            Parameters that define the simulations, e.g. seed or runtime
        """
        self.parameters = parameters
        self.hyper_parameters = hyper_parameters
        self.output = None

    def simulate(self, t_end: int = None, t_print: int = None,
                 num_firms: int = None, seed: int = None, seed_init: int = None,
                 run_in_period: int = None, save: str = None,
                 output_folder: str = 'output') -> pd.DataFrame:
        """ Simulate the Agent-based Model for a given seed. Optionally save its
        output as a txt file (tab separated).

        Parameters
        ----------
        t_end : int, default: None
            Total runtime of the model
        t_print : int, default: None
            At which interval to save outputs
        num_firms : int, default: None
            Number of firms in the simulation
        seed : int, default: 0
            Random seed of the simulation
        seed_init : int, default: None
            Random seed for the initial values of the simulation
        run_in_period : int, default: 0
            How many periods to cut off at the beginning of the simulation
        save : str, default: None
            filename to save the .txt output. 'mark0_centralbank' is prepended
        output_folder : str, default: 'output'
            where the output.txt is saved

        Returns
        -------
        output : pd.DataFrame
            time-series of the model simulation
        """
        # Adjust the hyperparameters if necessary
        if t_end:
            self.hyper_parameters['T'] = t_end
        if t_print:
            self.hyper_parameters['tprint'] = t_print
        if run_in_period:
            self.hyper_parameters['Teq'] = run_in_period
        if num_firms:
            self.hyper_parameters['N'] = num_firms
        if seed:
            self.hyper_parameters['seed'] = seed
        if seed_init:
            self.hyper_parameters['seed_init'] = seed_init

        # Decide whether the output .txt is temporary
        if save:
            save = self.model_name + '_' + save
        else:
            save = self.model_name + '_temp'

        # Name of save file and model call
        filename = os.sep.join([os.getcwd(), output_folder, save])
        function_call = self._model_call(filename)

        # Generate an output folder to store txt files
        if not os.path.isdir(output_folder):
            print('Created output folder')
            os.makedirs(output_folder)

        # Run the model
        os.system(function_call)
        # Retrieve output as a pandas DataFrame
        try:
            self.output = self._read_output(filename + '.txt')
            self.output = self.output.loc[:, self.variables]
        except pd.errors.EmptyDataError:
            self.output = pd.DataFrame(columns=self.variables)

        if not save:
            os.remove(filename + '.txt')
        return self.output.loc[:, self.variables]

    def plot_results(self, save: str = None):
        """ Plot the results of the Agent-based model

        Parameters
        ----------
        df: pd.DataFrame
            dataframe of the simulated timeseries
        save: str
            path where the figure should be saved. If not given,
            will show the figure
        """

        assert self.output is not None, "Simulate first"

        tmin, tmax = self.output.index[0], self.output.index[-1]
        fig, axs = plt.subplots(self.output.shape[1], 1)
        for i, col in enumerate(self.output.columns):
            axs[i].plot(self.output.loc[:, col])
            axs[i].set_title(self.output_series[col])
            axs[i].minorticks_on()
            axs[i].set_xlim(tmin, tmax)

        fig.set_size_inches(10, 20)
        fig.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight', format='pdf')
            plt.close()
        else:
            plt.show()

    def _model_call(self) -> str:
        """ Function to generate the command-line call for the ABM

        Returns
        -------
        call : str
            commandline argument that calls the model
        """
        raise NotImplementedError

    def _read_output(self, path: str, cutoff: int = 0) -> pd.DataFrame:
        """ Read tab separated values from txt file

        Parameters
        ----------
        path: str
            Location of the tab-separated .txt file of output

        Returns
        -------
        df: pd.DataFrame
            DataFrame of the simulation time-series
        """
        df = pd.read_csv(path, sep="\t", header=None, index_col=None)
        df = df.astype(float)
        assert df.shape[1] == len(self.output_series)
        df.columns = list(self.output_series.keys())
        df.set_index('t', inplace=True)     
        
        return df.loc[cutoff:, :]

    def _read_parameters(self, hyper: bool = False):
        """
        Read defaultparameters and hyper parameters from yaml files

        Parameters
        ----------
        hyper : bool, optional
            If true read hyper parameter, else read parameter. The default is False.

        Returns
        -------
        parameter: dict.
            Parameter dict from yaml file
        """
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = f'{file_path}/{self.model_name}/parameters/'
        if hyper:
            with open(path + 'default_hyper_parameters.yaml') as f:
                parameter = yaml.load(f)
        else:
            with open(path + 'default_parameters.yaml') as f:
                parameter = yaml.load(f)

        return parameter
