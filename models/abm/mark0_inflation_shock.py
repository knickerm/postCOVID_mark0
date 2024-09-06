"""
Interface to the mark0_covid_shock model that was originally developed
in C++ by Stanislao Gualdi. The original code can be found in /mark0_covid_shock
"""
__author__ = "Karl Naumann-Woleske, Max Sina Knicker"
__credits__ = ["Max Sina Knicker", "Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import os

import numpy as np
import pandas as pd
import yaml
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

from abm import AgentBasedModel
import warnings
warnings.filterwarnings('ignore', '.*yaml.load().*', )

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


class Mark0_INFLATION_SHOCK(AgentBasedModel):
    """ This class is the Python interface to the Mark0 COVID model
    dicussed in Bouchaud et al. (2021) "V –, U –, L – or W–shaped economic
    recovery after COVID-19: Insights from an Agent Based Model".

    The underlying methods are based on the authors' original code, written
    in C++ by Stanislao Gualdi. There have been minor modifications to this
    code, which can be found in the mark0_covid_shock/ directory.

    Attributes
    ----------
    call_path : str, default: ./models/mark0_centralbank/Debug/mark0_centralbank
        path to the mark0 .exe
    model_name : str, default: 'mark0_centralbank'
    base_call_func : str, default: './mark0'
        commandline call to activate the model
    output : pd.DataFrame
        dataframe of the time-series of output
    """

    def __init__(self, parameters: dict = None, hyper_parameters: dict = None,
                 variables: list = None,
                 base_call: str = 'mark0_inflation_shock', call_path: str = ''):
        """ Initialisation of the Mark0_COVID_SHOCK class. Sets the parameters
        as well as the call function.

        Parameters
        ----------
        parameters : dict, default: None
            Dictionary of the parameters used to call the Mark0-COVID
            model.
        hyper_parameters : dict, default: None
            Hyper-parameters for the model incl. runtime, number of firms, seed,
        variables: list[str], default: []
            List of outputs that should be considered in the returned outputs
        base_call : str, default: './mark0
            The baseline function to call in the terminal to run the mark0 model
        call_path : str, default: ''
            Location of the of the mark0_covid_shock.exe file
        """

        # Mark-0 COVID Output series
        self.output_series =  \
            {'t': r"Time",
             'u': r"Unemployment Rate (\%)",
             'bust': r"Bankruptcy Rate (\%)",
             'Pavg': r"Average Price $\bar{p}$",
             'Wavg': r"Average Wage $\bar{w}$",
             'S': r"Household Savings $S$",
             'Atot': r"Firm Balance $\sum_i\mathcal{E}_i$",
             'firm-savings': r"Firm deposits $\mathcal{E}^+$",
             'debt-tot': r"Firm liabilities $\mathcal{E}^-$",
             'inflation': r'Inflation rate $\pi$',
             'pi-avg': r"EMA Avg. Inflation $\tilde{\pi}$",
             'propensity': r"Consumption Rate $c$",
             'k': r"Debt-to-Equity Ratio",
             'Dtot': r"Total Demand $\sum_i D_i$",
             'rhom': r"Loan interest rate $\rho^l$",
             'rho': r"Central Bank Rate $\rho_0$",
             'rhop': r"Deposit interest rate $\rho^d$",
             'pi-used': r"Household inflation expectation $\hat{\pi}$",
             'tau_tar': r"Expect $\pi$ weight of CB target inflation",
             'tau_meas': r"Expect $\pi$ weight of EWMA inflation",
             'R': r"Hiring/firing rate",
             'Wtot': r"To be defined yet",
             'etap-avg': r"To be defined yet",
             'etam-avg': r"To be defined yet",
             'w-nominal': r"To be defined yet",
             'Ytot': r"Total Output Y",
             'deftot': r"To be defined yet",
             'profits': r"To be defined yet",
             'debt-ratio': r"To be defined yet",
             'firms-alive': r"To be defined yet",
             'left': r"To be defined yet",
             'u-af': r"Unemployment Rate (\%)",
             'bust-af':  r"Bankruptcy Rate (\%)",
             'frag': r"Financial Fragility $\Phi$",
             'true_end': r"To be defined yet",
             'theta': r"To be defined yet",
             'ytot-temp': r"Total Output Y",
             'min-ytot': r"Total Output Y"}

        # Mark-0 COVID Parameter Names in LaTeX format (for graphing)
        self.parameter_names = {
            'R0': r'Hiring/firing rate, $R$',
            'theta': r'Loan rate effect on $R$, $\alpha_\Gamma$',
            'Gamma0': r'Baseline gamma, $\Gamma_0$',
            'rho0': r'Adjustment ratio $\frac{\gamma_w}{\gamma_p}$',
            'alpha': r'Price-adjustment size, $\gamma_p$',
            'phi_pi': r'Baseline firing propensity, $\eta_0$',
            'alpha_e': r'Wage adjustment to inflation',

            'pi_target': r'Baseline interest rate, $\rho^\star$',
            'e_target': r'CB reaction to inflation, $\phi_\pi$',
            'tau_tar': r'CB reaction to employment $\phi_\varepsilon$',
            'wage_factor': r'CB inflation target, $\pi^\star$',
            'y0': r'CB unemployment target, $\hat{\varepsilon}^\star$',

            'gammap': r'Default threshold, $\Theta$',
            'eta0m': r'Bankruptcy effect on bank interest rates, $f$',

            'tau_r': r'Real rate influence on consumption, $\alpha_c$',
            'alpha_g': r'Baseline propensity to consume $c_0$',
            'zeta': r'Household intensity of choice, $\beta$',
            'kappa': r'Weight $\pi^\star$ on $\hat{\pi}$, $\tau^T$',

            'G0': r'Revival frequency, $\phi$',
            'phi': r'EWMA memory, $\omega$',
            'omega': r'Initial Production, $y_0$',
            'delta': r'Dividend Rate $\delta$',
            'gamma_e': r'Factor of price change',
            'delta_p': r'Price change',
            'alpha_i': r'Sensitivity of firm expectations to inflation',
            'omega_i': r'Moving average parameter for expectation',
            'gamma_cb': r'Factor of increased CB activity',
            'theta_cb': r'Activity threshold of CB pi_used - pi_target',
            'gamma_p': r'Price factor, $\gamma_p$',
            'shock_st': r'Shock strength',
            'delta_e': r'Dividends Energy sector $\delta_e$'
        }

        if variables == None:
            self.variables = list(self.output_series.keys())
            self.variables.remove('t')
        else:
            self.variables = variables

        self.phase_space = ()

        # Set the call-function passed to the command line
        if call_path == '':
            folder = os.path.dirname(os.path.abspath(__file__))
            self.call_path = folder + '/mark0_inflation_shock/' + base_call
        else:
            self.call_path = call_path
        self.model_name = 'mark0_inflation_shock'
        self.base_call_func = base_call

        # Check if we have all parameters and augment if necessary
        if parameters is None:
            parameters = self._read_parameters()
        else:
            for k, v in self._read_parameters().items():
                if k not in parameters:
                    parameters[k] = v

        # Check if we have all parameters and augment if necessary
        if hyper_parameters is None:
            hyper_parameters = self._read_parameters(hyper=True)
        else:
            for k, v in self._read_parameters(hyper=True).items():
                if k not in hyper_parameters:
                    hyper_parameters[k] = v

        # Initialize the model.py super class - this sets self.parameters
        super().__init__(parameters, hyper_parameters)

    def simulate(self, t_end: int = None, t_print: int = None,
                 num_firms: int = None, seed: int = None, seed_init: int = None,
                 run_in_period: int = None, save: str = None,
                 output_folder: str = 'output') -> pd.DataFrame:
        """ Run the Mark-0 COVID Model. Optionally save its output as a txt file (tab
        separated).

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
        if run_in_period:
            self.hyper_parameters['Teq'] = run_in_period
        if t_print:
            self.hyper_parameters['tprint'] = t_print
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

        # Run the Mark-0 Centralbank model
        os.system(function_call)
        print(function_call)
        # Retrieve output as a pandas DataFrame
        try:
            self.output = self._read_output(filename + '.txt')
            #self.output = np.log(self.output + 1e7)
        except pd.errors.EmptyDataError:
            self.output = pd.DataFrame(columns=self.variables)

        if not save:
            os.remove(filename + '.txt')
        return self.output.loc[:, self.variables]

    def plot_results(self, plot_end = 2120, is_shade = True, save: str = ''):
        '''
        Plots the evolution of the economy after the shock
        
        Parameters:
        res (pandas.DataFrame): DataFrame with output from simulation
        t_start (int): Start of the shock 
        t_end (int): End of the shock
        
        Returns
        Generates the plot inline. 
        '''
        t_start = self.hyper_parameters['t_start']
        t_end = self.hyper_parameters['t_end']
        col_inset = ['u', 'bust-af', 'rhol', 'rhod']
    
        start=t_start-10
        end = plot_end
        y = range(start-t_start, end-t_start)
        f, ax = plt.subplots(nrows = 2, ncols =3, figsize=(12, 4), dpi=200,)
        cols1 = ['min-ytot', 'u', 'Pavg', 'Wavg']
        cols2 = ['bust-af', 'frag', 'S']
        colnames1 = ['Total output', 'U-rate', 'Inflation/Wages']
        colnames2 = ['Bankruptcies', 'Fragility', 'Savings']
        print(start, end, 1+(self.output['Pavg'][start:end].max()-1)*12)
        for i in range(2):
            ax[0,i].plot(y,self.output[cols1[i]][start:end])
            ax[0,i].set_title(colnames1[i])
        ax[0,-1].plot(y, 1+(self.output['Pavg'][start:end]-1)*12, label = "1+ Year. Infl.")
        ax[0,-1].plot(y, self.output[cols1[-1]][start:end], label = "Real Wages")
        ax[0,-1].legend()
        ax[0,-1].set_title(colnames1[-1])
        for i in range(3):
            ax[1,i].plot(y, self.output[cols2[i]][start:end])
            ax[1,i].set_title(colnames2[i])
        plt.tight_layout()
        
        
        if is_shade:
            for axis in ax.flatten():
                axis.axvspan(0,t_end-t_start, facecolor='0.5', alpha=0.5)
        plt.suptitle("Evolution after shock", fontsize=16)
        plt.subplots_adjust(top=0.85)
        
        if save != '':
            plt.savefig(save, format='pdf')

    def plot_philips_curve(self, plot_end = 2120, save: str = ''):
        """ Plot philips curve (inflation and unemployment) over time. The 
        theory excepts an inverse relationship between inflation and 
        unemployment. The higher inflation, the lower unemployment


        Returns
        -------
        None.

        """
        assert self.output is not None, "Simulate first"

        format_dict = {
            'u': {'y_fmt': PercentFormatter(1.0)},
            'bust': {'y_fmt': PercentFormatter(1.0)},
            'Pavg': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'Wavg': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'S': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'Atot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'firm-savings': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'debt-tot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'inflation': {'y_fmt': PercentFormatter(1.0)},
            'pi-avg': {'y_fmt': PercentFormatter(1.0)},
            'propensity': {'y_fmt': PercentFormatter(1.0)},
            'k': {'y_fmt': FuncFormatter('{0:.2f}'.format)},
            'Dtot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'rhom': {'y_fmt': PercentFormatter(1.0)},
            'rho': {'y_fmt': PercentFormatter(1.0)},
            'rhop': {'y_fmt': PercentFormatter(1.0)},
            'pi-used': {'y_fmt': PercentFormatter(1.0)},
            'tau_tar': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'tau_meas': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'R': {'y_fmt': FuncFormatter('{0:.0f}'.format)}
        }

        t_start = self.hyper_parameters['t_start']
        start = t_start - 10
        end = plot_end
        
        tmin,tmax = self.output.index[0], self.output.index[-1]
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(self.output.u[start:end], 1+(self.output['Pavg'][start:end]-1)*12, 
                label='Philips Curve')
        ax.xaxis.set_major_formatter(format_dict['u']['y_fmt'])
        ax.yaxis.set_major_formatter(format_dict['Pavg']['y_fmt'])
        ax.set_title('Philips Curve', fontsize=16)
        ax.set_xlim(tmin, tmax)
        ax.set_xlim(0, 1)
        if save != '':
            plt.savefig(save, format='pdf')
        plt.show()

        return None

    def plot_unemployment(self):
        """ Plot unemployment over time


        Returns
        -------
        None.

        """
        assert self.output is not None, "Simulate first"

        format_dict = {
            'u': {'y_fmt': PercentFormatter(1.0)},
            'bust': {'y_fmt': PercentFormatter(1.0)},
            'Pavg': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'Wavg': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'S': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'Atot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'firm-savings': {'y_fmt': FuncFormatter('{0:f}'.format)},
            'debt-tot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'inflation': {'y_fmt': PercentFormatter(1.0)},
            'pi-avg': {'y_fmt': PercentFormatter(1.0)},
            'propensity': {'y_fmt': PercentFormatter(1.0)},
            'k': {'y_fmt': FuncFormatter('{0:.2f}'.format)},
            'Dtot': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'rhom': {'y_fmt': PercentFormatter(1.0)},
            'rho': {'y_fmt': PercentFormatter(1.0)},
            'rhop': {'y_fmt': PercentFormatter(1.0)},
            'pi-used': {'y_fmt': PercentFormatter(1.0)},
            'tau_tar': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'tau_meas': {'y_fmt': FuncFormatter('{0:.0f}'.format)},
            'R': {'y_fmt': FuncFormatter('{0:.0f}'.format)}
        }


        tmin,tmax = self.output.index[0], self.output.index[-1]
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(self.output.loc[:, 'u'], label='u')
        ax.yaxis.set_major_formatter(format_dict['u']['y_fmt'])
        ax.set_title(self.output_series['u'])
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, 1)
        plt.show()

        return None

    def detect_phase(self, xvariable: str = 'R', yvariable: str = 'theta'):
        """Find out in which phase the economy is
        Numbering according to "Tipping points in macroeconomic Agent-Based models".
        1 Full Unemployment (FU)
        2 Residual Unemployment (RU)
        3 Endogenous Crises (EC)
        4 Full Employment (FE)

        Returns
        -------
        tuple
        (variable 1, variable 2, phase transition)
        """

        if (self.output.loc[:,'u'] < 0.07).any():
            print("You are in Full Employment Phase")
            phase_space = (self.parameters[xvariable], self.parameters[yvariable], 4)
            if self.output.loc[:,'u'].std()>0.01:
                print("This could be a candidate for Endogenous Crisis Phase")
                phase_space = (self.parameters[xvariable], self.parameters[yvariable], 3)

        elif (self.output.loc[:,'u'] > 0.9).any():
            print("You are in Full Unemployment Phase")
            phase_space = (self.parameters[xvariable], self.parameters[yvariable], 1)

        else:
            print("You are in Residual Unemployment Phase")
            phase_space = (self.parameters[xvariable], self.parameters[yvariable], 2)

        return((phase_space,))

    def phase_classifier(self, cutoff: int = 500) -> str:
        """ Classify the phases into four different sections sequentially, that is
        pick the first one that applies:
        1. Full Employment (average employment in last cutoff periods < 10%)
        2. Full Unemployment (average unemployment in last cutoff periods > 90%)
        3. Endogenous Crises (std. dev. of employment in last cutoff periods > 0.1)
        4. Residual Unemployment (standard deviation of employment <)

        Parameters
        ----------
        unemployment : pd.Series
        cutoff : int (default 500)

        Returns
        -------
        phase : str (one of "FE", FU", "EC", "RU")
        """
        if self.output.loc[:, 'u'].shape[0] > cutoff:
            self.output.loc[:, 'u'] = self.output.loc[:, 'u'].iloc[-cutoff:]

        if self.output.loc[:, 'u'].mean() < 0.1:
            return "FE"
        elif self.output.loc[:, 'u'].mean() > 0.9:
            return "FU"
        elif self.output.loc[:, 'u'].std() > 0.1:
            return "EC"
        else:
            return "RU"

    def create_phase_diagram(self, xvariable: str = 'R', yvariable: str = 'theta', xmin: float = 0.0, xmax: float = 4.0,
                             ymin: float = 0.0, ymax: float = 3.0,
                             xinterval: float = 0.5, yinterval: float = 0.5):
        """Create phase diagram
        Set the the boundaries of the phase space

        Parameters
        ----------
        xvariable : str, optional
            First dimension of phase space. The default is 'R'.
        yvariable : str, optional
            Second dimension of phase space. The default is 'theta'.
        xmin : float, optional
            Lower boundary for xvariable. The default is 0.0.
        xmax : float, optional
            Upper boundary for xvariable. The default is 4.0.
        ymin : float, optional
            Lower boundary for yvariable. The default is 0.0.
        ymax : float, optional
            Upper boundary for yvariable . The default is 3.0.
        xinterval : float, optional
            Densitiy of measurement points within first dimension of phase space. The default is 0.5.
        yinterval : float, optional
            Densitiy of measurement points within second dimension of phase space. The default is 0.5.

        Returns
        -------
        None.

        """

        self.phase_space = ()
        for i in np.arange(xmin, xmax, xinterval):
            for j in np.arange(ymin, ymax, yinterval):
                self.parameters[xvariable] = i
                self.parameters[yvariable] = j
                print(xvariable, "=", self.parameters[xvariable], ", ", yvariable, "=", self.parameters[yvariable])

                self.simulate(t_end=int(2e3), run_in_period=500, save='test')
                phase = self.detect_phase(xvariable, yvariable)
                self.phase_space += phase
        self.plot_phase_diagram(xvariable, yvariable)

    def plot_phase_diagram(self, xvariable: str = 'R', yvariable: str = 'theta'):
        """Plot phase diagramm

        Parameters
        ----------
        phase_space : tuple
            Tuple generated by asser_phase, where first two entries spann the
            phase space and the third entry indicates the phase.

        Returns
        -------
        None.

        """


        fig, ax = plt.subplots()
        cdict = {1: 'black', 2: 'red', 3: 'green', 4: 'blue'}
        label = ['FU', 'RU', 'EC', 'FE']

        for g in np.unique(np.array(self.phase_space, dtype=int)[:,2]):
            ix = np.where(np.array(self.phase_space, dtype=int)[:,2] == g)
            ax.scatter(np.array(self.phase_space)[:,0][ix], np.array(self.phase_space)[:,1][ix], c = cdict[g], label = label[g-1], s = 100)

        plt.ylabel(yvariable.replace('_', ' '), fontsize=40)
        plt.xlabel(xvariable.replace('_', ' '), fontsize=40)
        ax.set_xticklabels(ax.get_xticks(), size = 25)
        ax.set_yticklabels(ax.get_yticks(), size = 25)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(prop={'size': 30})
        plt.show()

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
        #print(path)
        df = pd.read_csv(path, sep="\t", header=None, index_col=False)
        
        assert df.shape[1] == len(self.output_series)
        df.columns = self.output_series
        df.set_index('t', inplace=True)

        if cutoff == 0:
            cutoff = -self.hyper_parameters['Teq']
        return df.loc[cutoff:, :]


    def _model_call(self, output_file: str = '_temp') -> str:
        """ Generate the function call for Mark-0 Covid

        Parameters
        ----------
        output_file : str, default: '_temp'
            Name of the output file that will be saved as .txt by program

        Returns
        -------
        call: str
            commandline call for Mark-0
        """
        order_hyper = ('zfactor', 'cfactor', 'seed', 'shockflag', 't_start', 
                       't_end', 'policy_start', 'policy_end', 'helico', 'N',
                       'extra_cons', 'adapt', 'extra_steps', 'T', 'Teq', 
                       'tprint', 'renorm', 'cbon', 'tprod',
                       'price_start', 'price_end', 'kappa_h', 'delta_w')
        items = [self.call_path,
                 *[f"{self.parameters[p]}" for p in self.parameter_names],
                 *[f"{self.hyper_parameters[v]}" for v in order_hyper],
                 output_file]
        return ' '.join(items)

    def _read_parameters(self, hyper: bool = False):
        """
        Read default parameters and hyper parameters from yaml files

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
                parameter = yaml.load(f, Loader=yaml.FullLoader)
        else:
            with open(path + '/default_parameters_noCB.yaml') as f:
                parameter = yaml.load(f, Loader=yaml.FullLoader)

        return parameter

    def _compile(self):
        """ Compile the C++ source on Ubuntu 20.04 with g++
        """
        # Linux
        call = "g++ mark0_covid_shock.cpp -l:libgsl.so.23.1.0 -o mark0_covid_shock"
        # Mac
        call = "g++ mark0_covid_shock.cpp -L/usr/local/Cellar/ -lgsl -o mark0_covid_shock"
        # Windows
        call1 = r'g++ -c .\mark0_covid_shock.cpp -I"C:\Program Files (x86)\GnuWin32\include" -Wall'
        call2 = r'g++ -static mark0_covid_shock.o -L"C:\Program Files (x86)\GnuWin32\lib" -lgsl -lgslcblas -lm -o mark0_covid_shock'

        os.system(call)


if __name__ == "__main__":

    file_path = os.path.dirname(os.path.abspath(__file__))
    param_path = f'{file_path}/mark0_inflation/parameters/'


    fig_4 = ['no_policy']#, 'naive_policy', 'naive_policy_helico', 
             #'adaptive_policy']
    
    for policy in fig_4:
        with open(param_path + '/default_parameters_adding.yaml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    
        with open(param_path + f'default_hyper_parameters_{policy}.yaml') as f:
            hyper_params = yaml.load(f, Loader=yaml.FullLoader)
        
        
        m0covid = Mark0_INFLATION_SHOCK(params, hyper_params)
        tmp = m0covid.simulate(save='test')
        folder = '../../figures/inflation/paper_fig_4/'
        #m0covid.plot_results(save = folder + 'expectation_0.5_' + 
        #                            f'{policy}_c_shock_{ hyper_params["cfactor"] }' +
        #                            f'_prod_shock_{hyper_params["zfactor"]}_' +
        #                            f'shockflag_{hyper_params["shockflag"]}.pdf')
        m0covid.plot_philips_curve()
        #m0covid.plot_unemployment()
    #m0covid.detect_phase()
    #print(m0covid.phase_classifier())
    # m0covid.create_phase_diagram()
