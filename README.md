# Post-COVID Inflation & the Monetary Policy Dilemma: An Agent-Based Scenario Analysis

This repository contains the code used for the paper "Post-COVID Inflation and the Monetary Policy Dilemma: An Agent-Based Scenario Analysis". 

## Reference
- Knicker et al. (2024) *Post-COVID Inflation & the Monetary Policy Dilemma: An Agent-Based Scenario Analysis* ([Link to Paper](https://link.springer.com/article/10.1007/s11403-024-00413-3))

## Usage & Structure
The Mark0 model (written in C++) can be found in the `models/` directory. For the model there is a `.py` file, `mark0_inflation_shock.py`, that is the Python interface for the model. 

The Jupyter notebook file `run_model.ipynb` executes the model and generates the main figures from the paper. It can also serve as an example for running other experiments with the Mark0 model. To conduct different experiments, simply modify the parameter dictionary within the jupyter notebook. There is no need to alter the C++ code or any other configuration files.

The python file `plotting.py` entails code for formatting and plotting of the figures.

## Compilation of C++ code
The C++ code in the current repository is pre-compiled for Windows and can be executed (tested). To compile the code on other systems, use the following command in the terminal:

*Linux*
	
	g++ mark0_inflation_shock.cpp -l:libgsl.so.23.1.0 -o mark0_inflation_shock
	
*Mac*
	
	g++ mark0_inflation_shock.cpp -L/usr/local/Cellar/ -lgsl -o mark0_inflation_shock
	
*Windows*
	
	g++ -c .\mark0_inflation_shock.cpp -I"C:\Program Files (x86)\GnuWin32\include" -Wall
        g++ -static mark0_inflation_shock.o -L"C:\Program Files (x86)\GnuWin32\lib" -lgsl -lgslcblas -lm -o mark0_inflation_shock
	
## Contact
For any code related issue, please open an issue here or do not hesitate to send an email to
- Max Sina Knicker [max.knicker@cdtm.de](mailto:max.knicker@cdtm.de)

