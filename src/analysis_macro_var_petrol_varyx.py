# VARX but all combinations or Cholesky Ordering considered

import pandas as pd
import matplotlib
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    est_varx, manual_irf_subplots
from datetime import timedelta, date
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast
import itertools
import multiprocessing as mp

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

# III.A --- Set up parameters
# Base columns
cols_all_endog = ['ron95', 'pc', 'gfcf', 'gdp', 'neer', 'cpi', 'mgs10y']  # same as suah (2022) + ron95
cols_exog = ['gepu', 'maxgepu', 'brent']  # spin on the MU-VAR
# Generate all possible permutations as a list of lists (permutations returns a list of tuples)
cols_all_endog_perm = [list(i) for i in itertools.permutations(cols_all_endog)]
# drop NA in data
df = df[cols_all_endog + cols_exog]
df = df.dropna(axis=0)

# III.B --- Estimate models
tab_irf_all = pd.DataFrame(columns=['order', 'shock', 'response', 'horizon', 'irf'])
tab_irf_narrative = pd.DataFrame(columns=['order', 'shock', 'response', 'horizon', 'irf'])
tab_irf_narrative2 = pd.DataFrame(columns=['order', 'shock', 'response', 'horizon', 'irf'])
list_endog_narrative_fit = []  # store list of cholesky orders that satisfy the narrative sign restrictions
list_endog_narrative2_fit = []  # store list of cholesky orders that satisfy the narrative sign restrictions
for endog in tqdm(cols_all_endog_perm):
    # Estimate model
    res, irf = est_varx(
        df=df,
        cols_endog=endog,
        run_varx=True,
        cols_exog=cols_exog,
        choice_ic='hqic',
        choice_trend='c',
        choice_horizon=choice_horizon,
        choice_maxlags=1  # RON95 starts at 2014/15, shorter maxlag is needed
    )
    # Extract IRFs
    tab_irf = pd.DataFrame(columns=['shock', 'response', 'horizon', 'irf'])
    round_shock = 0
    for shock in endog:
        round_response = 0
        for response in endog:
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
    tab_irf['order'] = ' -> '.join(endog)  # cholesky ordering of the particular loop
    # Convert Q to A
    tab_irf_a = tab_irf.copy()  # work on deep copy so that quarterly version is saved
    tab_irf_a['horizon_year'] = tab_irf_a['horizon'] // 4
    tab_irf_a = tab_irf_a.groupby(['shock', 'response', 'horizon_year'])['irf'].mean().reset_index()
    # Parse those that satisfy 'the narrative'
    tab_irf_a0 = tab_irf_a[tab_irf_a['horizon_year'] == 0].copy()  # keep only 1st year
    # narrative: oil price = cost-push shock
    cond_petrol_costpush = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'ron95') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'ron95') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'ron95') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'ron95') & (tab_irf_a0['response'] == 'cpi')
                    , 'irf'].iloc[0] > 0
        )
    )
    # narrative: 1st year impact of cpi shock on gdp, pc, and gfcf is negative (cost push inflation)
    cond_costpush = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'cpi') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'cpi') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'cpi') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] < 0
        )
    )
    # narrative: 1st year impact of mgs10y shock on gdp, pc, gfcf, cpi is negative, and positive for neer (mp channel)
    cond_mp = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'mgs10y') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'mgs10y') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'mgs10y') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'mgs10y') & (tab_irf_a0['response'] == 'cpi')
                    , 'irf'].iloc[0] < 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'mgs10y') & (tab_irf_a0['response'] == 'neer')
                    , 'irf'].iloc[0] > 0
        )
    )
    # narrative: 1st year impact of pc shock on cpi are positive (demand pull)
    cond_demandpull_pc = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'pc') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'pc') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'pc') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'pc') & (tab_irf_a0['response'] == 'cpi')
                    , 'irf'].iloc[0] > 0
        )
    )
    # narrative: 1st year impact of gfcf shock on cpi are positive (demand pull)
    cond_demandpull_gfcf = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gfcf') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gfcf') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gfcf') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gfcf') & (tab_irf_a0['response'] == 'cpi')
                    , 'irf'].iloc[0] > 0
        )
    )
    # narrative: 1st year impact of gdp shock on cpi are positive (demand pull)
    cond_demandpull_gdp = bool(
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gdp') & (tab_irf_a0['response'] == 'pc')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gdp') & (tab_irf_a0['response'] == 'gfcf')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gdp') & (tab_irf_a0['response'] == 'gdp')
                    , 'irf'].iloc[0] > 0
        ) and
        (
                tab_irf_a0.loc[
                    (tab_irf_a0['shock'] == 'gdp') & (tab_irf_a0['response'] == 'cpi')
                    , 'irf'].iloc[0] > 0
        )
    )
    # Consolidate into kitchen sink frame
    tab_irf_all = pd.concat([tab_irf_all, tab_irf], axis=0)  # top down
    # Consolidate into narrative frame
    if cond_petrol_costpush:
        # and cond_costpush:  # and cond_demandpull_pc and cond_demandpull_gfcf and cond_demandpull_gfcf:
        # and cond_mp:
        tab_irf_narrative = pd.concat([tab_irf_narrative, tab_irf], axis=0)  # top down
        list_endog_narrative_fit = list_endog_narrative_fit + [' -> '.join(endog)]
    if cond_costpush:
        # and cond_demandpull_pc and cond_demandpull_gfcf and cond_demandpull_gfcf:
        # and cond_mp:
        tab_irf_narrative2 = pd.concat([tab_irf_narrative2, tab_irf], axis=0)  # top down
        list_endog_narrative2_fit = list_endog_narrative2_fit + [' -> '.join(endog)]
# Collapse kitchen sink and narrative frames
tab_irf_all_avg = tab_irf_all.groupby(['shock', 'response', 'horizon'])['irf'].mean().reset_index()
tab_irf_narrative_avg = tab_irf_narrative.groupby(['shock', 'response', 'horizon'])['irf'].mean().reset_index()
tab_irf_narrative2_avg = tab_irf_narrative2.groupby(['shock', 'response', 'horizon'])['irf'].mean().reset_index()
# Generate output
tab_irf_all.to_parquet(path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_' + input_suffix + '.parquet')
tab_irf_all.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_' + input_suffix + '.csv', index=False)
tab_irf_all_avg.to_parquet(path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_avg_' + input_suffix + '.parquet')
tab_irf_all_avg.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_avg_' + input_suffix + '.csv',
                       index=False)
tab_irf_narrative.to_parquet(path_output + 'macro_var_petrol_irf_all_varyx_narrative_' + input_suffix + '.parquet')
tab_irf_narrative.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_narrative_' + input_suffix + '.csv', index=False)
tab_irf_narrative_avg.to_parquet(
    path_output + 'macro_var_petrol_irf_all_varyx_narrative_avg_' + input_suffix + '.parquet')
tab_irf_narrative_avg.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_narrative_avg_' + input_suffix + '.csv',
                             index=False)
tab_irf_narrative2.to_parquet(path_output + 'macro_var_petrol_irf_all_varyx_narrative2_' + input_suffix + '.parquet')
tab_irf_narrative2.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_narrative2_' + input_suffix + '.csv',
                          index=False)
tab_irf_narrative2_avg.to_parquet(
    path_output + 'macro_var_petrol_irf_all_varyx_narrative2_avg_' + input_suffix + '.parquet')
tab_irf_narrative2_avg.to_csv(path_output + 'macro_var_petrol_irf_all_varyx_narrative2_avg_' + input_suffix + '.csv',
                              index=False)
# Count number of cholesky ordering that satisfy the narrative sign restrictions
print('\n----- Number of Cholesky orders that satisfy the specified narrative sign restrictions: '
      + str(len(list_endog_narrative_fit)) + ' -----')

print('\n----- Number of Cholesky orders that satisfy the specified narrative sign restrictions: '
      + str(len(list_endog_narrative2_fit)) + ' -----')

# III.C --- Plot combined IRFs
# Kitchen sink
fig_kitchensink_avg = manual_irf_subplots(
    data=tab_irf_all_avg,
    endog=cols_all_endog,
    shock_col='shock',
    response_col='response',
    irf_col='irf',
    horizon_col='horizon',
    main_title='All Possible Cholesky Orders (With RON95): Average IRFs' + ' (' + str(
        len(cols_all_endog_perm)) + ' Runs)',
    maxrows=len(cols_all_endog),
    maxcols=len(cols_all_endog),
    line_colour='black',
    annot_size=14,
    font_size=14
)
fig_kitchensink_avg.write_image(path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_avg_' + input_suffix + '.png')
telsendimg(
    conf=tel_config,
    path=path_output + 'macro_var_petrol_irf_all_varyx_kitchensink_avg_' + input_suffix + '.png',
    cap='macro_var_petrol_irf_all_varyx_kitchensink_avg_' + input_suffix
)
# Narrative sign restrictions
fig_narrative_avg = manual_irf_subplots(
    data=tab_irf_narrative_avg,
    endog=cols_all_endog,
    shock_col='shock',
    response_col='response',
    irf_col='irf',
    horizon_col='horizon',
    main_title='All Cholesky Orders Satisfying Narrative Sign Restrictions (With RON95): Average IRFs'
               + ' (' + str(len(list_endog_narrative_fit)) + ' out of ' + str(len(cols_all_endog_perm)) + ' Runs)',
    maxrows=len(cols_all_endog),
    maxcols=len(cols_all_endog),
    line_colour='black',
    annot_size=14,
    font_size=14
)
fig_narrative_avg.write_image(path_output + 'macro_var_petrol_irf_all_varyx_narrative_avg_' + input_suffix + '.png')
telsendimg(
    conf=tel_config,
    path=path_output + 'macro_var_petrol_irf_all_varyx_narrative_avg_' + input_suffix + '.png',
    cap='macro_var_petrol_irf_all_varyx_narrative_avg_' + input_suffix
)
# Narrative sign restrictions 2
fig_narrative2_avg = manual_irf_subplots(
    data=tab_irf_narrative2_avg,
    endog=cols_all_endog,
    shock_col='shock',
    response_col='response',
    irf_col='irf',
    horizon_col='horizon',
    main_title='All Cholesky Orders Satisfying Narrative Sign Restrictions 2 (With RON95): Average IRFs'
               + ' (' + str(len(list_endog_narrative2_fit)) + ' out of ' + str(len(cols_all_endog_perm)) + ' Runs)',
    maxrows=len(cols_all_endog),
    maxcols=len(cols_all_endog),
    line_colour='black',
    annot_size=14,
    font_size=14
)
fig_narrative2_avg.write_image(path_output + 'macro_var_petrol_irf_all_varyx_narrative2_avg_' + input_suffix + '.png')
telsendimg(
    conf=tel_config,
    path=path_output + 'macro_var_petrol_irf_all_varyx_narrative2_avg_' + input_suffix + '.png',
    cap='macro_var_petrol_irf_all_varyx_narrative2_avg_' + input_suffix
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_macro_var_petrol_varyx: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
