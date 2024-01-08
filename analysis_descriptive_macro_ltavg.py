import pandas as pd
import matplotlib
from helper import \
    telsendmsg, telsendimg, telsendfiles, \
    est_varx, manual_irf_subplots
from datetime import timedelta, date
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast
import itertools
from tabulate import tabulate

time_start = time.time()

# 0 --- Main settings
load_dotenv()
choice_horizon = int(os.getenv('IRF_HORIZON'))
choice_seed = int(os.getenv('RNG_SEED'))
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/macro/'
path_output = './output/'
macro_qoq = ast.literal_eval(os.getenv('MACRO_QOQ'))
macro_yoy = ast.literal_eval(os.getenv('MACRO_YOY'))
macro_ln_levels = ast.literal_eval(os.getenv('MACRO_LN_LEVELS'))
macro_ln_qoq = ast.literal_eval(os.getenv('MACRO_LN_QOQ'))
macro_ln_yoy = ast.literal_eval(os.getenv('MACRO_LN_YOY'))
if not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    input_suffix = 'levels'
if macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    input_suffix = 'qoq'
if macro_yoy and not macro_qoq and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    input_suffix = 'yoy'
if macro_ln_levels and not macro_yoy and not macro_qoq and not macro_ln_qoq and not macro_ln_yoy:
    input_suffix = 'ln_levels'
if macro_ln_qoq and not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_yoy:
    input_suffix = 'ln_qoq'
if macro_ln_yoy and not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq:
    input_suffix = 'ln_yoy'

# I --- Load data
df = pd.read_parquet(path_data + 'macro_data_' + input_suffix + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('q')

# II --- Pre-analysis cleaning
# Set index
df = df.set_index('quarter')

# III --- Set up parameters
# Base columns
cols_all_endog = ['pc', 'gfcf', 'gdp', 'neer', 'cpi', 'mgs10y']  # same as suah (2022)
cols_exog = ['gepu', 'maxgepu', 'brent']  # spin on the MU-VAR
# Generate all possible permutations as a list of lists (permutations returns a list of tuples)
cols_all_endog_perm = [list(i) for i in itertools.permutations(cols_all_endog)]
# drop NA in data
df = df[cols_all_endog + cols_exog]
df = df.dropna(axis=0)

# IV --- Compute long-term averages
# Restrict time
df = df[(df.index >= '1Q2015') & (df.index <= '4Q2019')]
# Tabulate averages
tab_mean = pd.DataFrame(df.mean()).round(3)
print(
    tabulate(
        tabular_data=tab_mean,
        showindex=True,
        headers='keys',
        tablefmt="pretty"
    )
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_macro_ltavg: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
