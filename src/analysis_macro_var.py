import pandas as pd
import matplotlib
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    est_varx
from datetime import timedelta, date
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

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

# III.A --- Analysis
# Determine order
cols_order = ['pc', 'gfcf', 'gdp', 'neer', 'cpi', 'mgs10y']  # same as suah (2022)
cols_exog = ['gepu', 'maxgepu', 'brent']  # spin on the MU-VAR
# Estimate model
# matplotlib.use('TKAgg')
res, irf = est_varx(
    df=df,
    cols_endog=cols_order,
    run_varx=True,
    cols_exog=cols_exog,
    choice_ic='hqic',
    choice_trend='c',
    choice_horizon=choice_horizon,
    choice_maxlags=4
)
# OIRF
fig_irf = irf.plot(
    orth=True,
    signif=0.95,
    subplot_params=dict(fontsize=10),
    seed=choice_seed
)
fig_irf.savefig(fname=path_output + 'macro_var_irf_all_' + input_suffix + '.png')
telsendimg(
    conf=tel_config,
    path=path_output + 'macro_var_irf_all_' + input_suffix + '.png',
    cap='macro_var_irf_all_' + input_suffix
)
# COIRF
fig_cirf = irf.plot_cum_effects(
    orth=True,
    signif=0.95,
    subplot_params=dict(fontsize=10),
    seed=choice_seed
)
fig_cirf.savefig(fname=path_output + 'macro_var_cirf_all_' + input_suffix + '.png')
# telsendimg(
#     conf=tel_config,
#     path=path_output + 'macro_var_cirf_all_' + input_suffix + '.png',
#     cap='macro_var_cirf_all_' + input_suffix
# )
# Generate IRF table manually (arrays are grouped by responses)
tab_irf = pd.DataFrame(columns=['shock', 'response', 'horizon', 'irf'])
round_shock = 0
for shock in tqdm(cols_order):
    round_response = 0
    for response in cols_order:
        round_horizon = 0
        for horizon in range(0, choice_horizon + 1):
            irf_interim = irf.orth_irfs[horizon][round_response][round_shock]  # single value extracted
            tab_irf_interim = pd.DataFrame(
                {
                    'shock': [shock],
                    'response': [response],
                    'horizon': [horizon],
                    'irf': [irf_interim]
                }
            )
            tab_irf = pd.concat([tab_irf, tab_irf_interim], axis=0)  # top-down
            round_horizon += 1
        round_response += 1
    round_shock += 1
tab_irf = tab_irf.reset_index(drop=True)

# Output all IRFs as table
tab_irf.to_parquet(path_output + 'macro_var_irf_all_' + input_suffix + '.parquet')
tab_irf.to_csv(path_output + 'macro_var_irf_all_' + input_suffix + '.csv', index=False)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_macro_var: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
