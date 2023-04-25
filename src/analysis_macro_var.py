import pandas as pd
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    est_varx
from datetime import timedelta, date
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/macro/'
macro_qoq = ast.literal_eval(os.getenv('MACRO_QOQ'))
macro_yoy = ast.literal_eval(os.getenv('MACRO_YOY'))
if not macro_qoq and not macro_yoy:
    input_suffix = 'levels'
if macro_qoq and not macro_yoy:
    input_suffix = 'qoq'
if not macro_qoq and macro_yoy:
    input_suffix = 'yoy'

# I --- Load data
df = pd.read_parquet(path_data + 'macro_data_' + input_suffix + '.parquet')
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('q')

# II --- Pre-analysis cleaning
# Set index
df = df.set_index('quarter')

# III --- Analysis
# Determine order
cols_order = ['pc', 'gfcf', 'gdp', 'neer', 'cpi', 'gepu', 'mgs10y']
cols_exog = ['maxgepu', 'brent']
# Estimate model
res, irf = est_varx(
    df=df,
    cols_endog=cols_order,
    run_varx=True,
    cols_exog=cols_exog,
    choice_ic='hqic',
    choice_trend='c',
    choice_horizon=16,
    choice_maxlags=4
)

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')