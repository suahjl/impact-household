import pandas as pd
import numpy as np
from tabulate import tabulate
from src.helper import \
    telsendmsg, telsendimg, telsendfiles
import dataframe_image as dfi
from datetime import timedelta, date
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
# Paths and script-specific settings
load_dotenv()
choice_horizon = int(os.getenv('IRF_HORIZON'))
tel_config = os.getenv('TEL_CONFIG')
path_output = './output/'

# Which macro-analysis
macro_qoq = ast.literal_eval(os.getenv('MACRO_QOQ'))
macro_yoy = ast.literal_eval(os.getenv('MACRO_YOY'))
macro_ln_levels = ast.literal_eval(os.getenv('MACRO_LN_LEVELS'))
macro_ln_qoq = ast.literal_eval(os.getenv('MACRO_LN_QOQ'))
macro_ln_yoy = ast.literal_eval(os.getenv('MACRO_LN_YOY'))
if not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    macro_suffix = 'levels'
if macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    macro_suffix = 'qoq'
if macro_yoy and not macro_qoq and not macro_ln_levels and not macro_ln_qoq and not macro_ln_yoy:
    macro_suffix = 'yoy'
if macro_ln_levels and not macro_yoy and not macro_qoq and not macro_ln_qoq and not macro_ln_yoy:
    macro_suffix = 'ln_levels'
if macro_ln_qoq and not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_yoy:
    macro_suffix = 'ln_qoq'
if macro_ln_yoy and not macro_qoq and not macro_yoy and not macro_ln_levels and not macro_ln_qoq:
    macro_suffix = 'ln_yoy'

# Which micro-analysis
income_choice = os.getenv('INCOME_CHOICE')
outcome_choice = os.getenv('OUTCOME_CHOICE')
use_first_diff = ast.literal_eval(os.getenv('USE_FIRST_DIFF'))
if use_first_diff:
    fd_suffix = 'fd'
elif not use_first_diff:
    fd_suffix = 'level'
show_ci = ast.literal_eval(os.getenv('SHOW_CI'))
hhbasis_adj_analysis = ast.literal_eval(os.getenv('HHBASIS_ADJ_ANALYSIS'))
equivalised_adj_analysis = ast.literal_eval(os.getenv('EQUIVALISED_ADJ_ANALYSIS'))
if hhbasis_adj_analysis:
    hhbasis_suffix = '_hhbasis'
    hhbasis_chart_title = ' (Total HH)'
if equivalised_adj_analysis:
    hhbasis_suffix = '_equivalised'
    hhbasis_chart_title = ' (Equivalised)'
elif not hhbasis_adj_analysis and not equivalised_adj_analysis:
    hhbasis_suffix = ''
    hhbasis_chart_title = ''
show_ci = ast.literal_eval(os.getenv('SHOW_CI'))
hhbasis_cohorts_with_hhsize = ast.literal_eval(os.getenv('HHBASIS_COHORTS_WITH_HHSIZE'))

# Which parameters
choice_macro_shock = 'Private Consumption'
choice_micro_outcome = 'Consumption'
choices_macro_response = ['GDP', 'Private Consumption', 'CPI', 'NEER', 'MGS10Y']

# I --- Load parameter estimates
irf = pd.read_parquet(path_output + 'macro_var_irf_all_varyx_narrative_avg_' + macro_suffix + '.parquet')
# Nice names
dict_cols_nice = {
    'ex': 'Exports',
    'gc': 'Public Consumption',
    'pc': 'Private Consumption',
    'gfcf': 'Investment',
    'im': 'Imports',
    'gdp': 'GDP',
    'brent': 'Brent',
    'cpi': 'CPI',
    'gepu': 'Uncertainty',
    'myrusd': 'MYR/USD',
    'mgs10y': 'MGS 10-Year Yields',
    'klibor1m': 'KLIBOR 1-Month',
    'neer': 'NEER',
    'reer': 'REER',
    'klibor3m': 'KLIBOR 3-Month',
    'importworldipi': 'Import-Weighted World IPI',
    'prodworldipi': 'Production-Weighted World IPI',
    'maxgepu': 'Uncertainty Shocks'
}
for i in ['shock', 'response']:
    irf[i] = irf[i].replace(dict_cols_nice)
mpc = pd.read_parquet(path_output +
                      'params_table_fe_consol_strat_cons_' +
                      income_choice + '_' + fd_suffix + '_subgroups' + hhbasis_suffix + '.parquet').reset_index()

# II --- Parse parameter estimates
irf = irf[irf['shock'] == choice_macro_shock]
mpc = mpc[mpc['outcome_variable'] == choice_micro_outcome]

# III.0 --- Compile eligible N(HHs)
n_pop = 32447385
n_citizens = 29756315
n_15_above = 24675545
n_15_59 = 21341441
n_60_above = 3334104
n_hh = 8234644
ngdp_2022 = 1788184000000
rgdp_2022 = 1507305000000
npc_2022 = 1029952000000
rpc_2022 = 907573000000
deflator_2022 = 118.65
deflator_2023 = 118.65
pc_deflator_2022 = 113.48
pc_deflator_2023 = 113.48
dict_eligible_count = {
    'HHs: All': n_hh,
    'HHs: B40': 0.4 * n_hh,
    'HHs: B40 & M20-': 0.6 * n_hh,
    'HHs: B40 & M40': 0.8 * n_hh,
    'HHs: 0 Kid': (44.458933515226896 / 100) * n_hh,
    'HHs: 1 Kid': (19.829073546523293 / 100) * n_hh,
    'HHs: 2 Kids': (17.47983790073426 / 100) * n_hh,
    'HHs: 3 Kids': (10.825342053524857 / 100) * n_hh,
    'HHs: 4 Kids': (4.943225133410905 / 100) * n_hh,
    'HHs: 5 Kids': (1.6992336396099987 / 100) * n_hh,
    'HHs: 6 Kids': (0.5416683384825262 / 100) * n_hh,
    'HHs: 7+ Kids': (0.22268587248726077 / 100) * n_hh,
    'Individuals: Working Age Adults (15-59)': n_15_59,
    'Individuals: Elderly (60+)': n_60_above
}
eligible_count = pd.DataFrame(dict_eligible_count.items(), columns=['group', 'n_eligible'])


# III.A --- Compute landing impact using estimated MPCs
# Function

def create_transfers_layer(
        layer_name,
        amount_monthly,
        n_eligible_layer,
        mpc_estimate
):
    # compute
    amount_annual = amount_monthly * 12
    total_injection = n_eligible_layer * amount_annual
    total_injection_gdp = 100 * (total_injection / ngdp_2022)
    total_spent = mpc_estimate * total_injection
    gdp_impact = 100 * ((total_spent / (deflator_2023 / 100)) / rgdp_2022)
    pc_impact = 100 * ((total_spent / (pc_deflator_2023 / 100)) / rpc_2022)
    # piece together
    landing_interim = pd.DataFrame(
        {
            'Policy Layer': [layer_name],
            'Monthly Transfers (RM)': [amount_monthly],
            'Annual Transfers (RM)': [amount_annual],
            'Eligible HHs / Individuals': [n_eligible_layer],
            'Total Injection (RM bil)': [total_injection / 1000000000],
            'Total Injection (% of 2022 GDP)': [total_injection_gdp],
            'MPC from Transfers': [mpc_estimate],
            'Total Spent (RM bil)': [total_spent / 1000000000],
            'Impact on 2023 GDP Growth (pp)': [gdp_impact],
            'Impact on 2023 PC Growth (pp)': [pc_impact]
        }
    )
    # output
    return landing_interim


# Set up layers
list_layers = [
    'Universal',
    'B60',
    '1 Kid', '2 Kids', '3 Kids', '4 Kids', '5 Kids', '6 Kids', '7+ Kids',
    'Working Age', 'Elderly'
]
list_amount_monthly = [
    100,
    100,
    100, 200, 300, 400, 500, 600, 700,
    50, 100
]
list_mpc = [
    mpc.loc[mpc['subgroup'] == 'All HHs', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == 'B40 & M20-', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == '1 Kid or More', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == 'With Adult (18-59) HH Head', 'Parameter'].reset_index(drop=True)[0],
    mpc.loc[mpc['subgroup'] == 'With Elderly (60+) HH Head', 'Parameter'].reset_index(drop=True)[0],
]
list_eligible = [
    eligible_count.loc[eligible_count['group'] == 'HHs: All', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: B40 & M20-', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 1 Kid', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 2 Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 3 Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 4 Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 5 Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 6 Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'HHs: 7+ Kids', 'n_eligible'].reset_index(drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'Individuals: Working Age Adults (15-59)', 'n_eligible'].reset_index(
        drop=True)[0],
    eligible_count.loc[eligible_count['group'] == 'Individuals: Elderly (60+)', 'n_eligible'].reset_index(drop=True)[0],
]
# Compute all layers
landing = pd.DataFrame(
    columns=[
        'Policy Layer',
        'Monthly Transfers (RM)',
        'Annual Transfers (RM)',
        'Eligible HHs / Individuals',
        'Total Injection (RM bil)',
        'Total Injection (% of 2022 GDP)',
        'MPC from Transfers',
        'Total Spent (RM bil)',
        'Impact on 2023 GDP Growth (pp)',
        'Impact on 2023 PC Growth (pp)',
    ]
)
for layer, amount, eligible, mpc in tqdm(zip(list_layers, list_amount_monthly, list_eligible, list_mpc)):
    landing_interim = create_transfers_layer(
        layer_name=layer,
        amount_monthly=amount,
        n_eligible_layer=eligible,
        mpc_estimate=mpc
    )
    landing = pd.concat([landing, landing_interim], axis=0)  # top down
landing_total = pd.DataFrame(
    {
        'Policy Layer': ['Total'],
        'Monthly Transfers (RM)': [-1],
        'Annual Transfers (RM)': [-1],
        'Eligible HHs / Individuals': [-1],
        'Total Injection (RM bil)': [landing['Total Injection (RM bil)'].sum()],
        'Total Injection (% of 2022 GDP)': [landing['Total Injection (% of 2022 GDP)'].sum()],
        'MPC from Transfers': [-1],
        'Total Spent (RM bil)': [landing['Total Spent (RM bil)'].sum()],
        'Impact on 2023 GDP Growth (pp)': [landing['Impact on 2023 GDP Growth (pp)'].sum()],
        'Impact on 2023 PC Growth (pp)': [landing['Impact on 2023 PC Growth (pp)'].sum()]
    }
)
landing = pd.concat([landing, landing_total], axis=0)
landing = landing.reset_index(drop=True)
for i in ['Eligible HHs / Individuals']:
    landing[i] = landing[i].round(0).astype('int')
for i in ['Total Injection (RM bil)', 'Total Injection (% of 2022 GDP)',
          'Total Spent (RM bil)', 'Impact on 2023 GDP Growth (pp)', 'Impact on 2023 PC Growth (pp)']:
    landing[i] = landing[i].round(2)
for i in ['MPC from Transfers']:
    landing[i] = landing[i].round(2)
landing = landing.replace({-1: '-'})
print(
    tabulate(
        tabular_data=landing,
        showindex=False,
        headers='keys',
        tablefmt="pretty"
    )
)
# Export landing impact
landing = landing.astype('str')
landing.to_parquet(path_output + 'landing_impact_sim_' +
                   income_choice + '_' + outcome_choice + '_' +
                   fd_suffix + hhbasis_suffix + '.parquet')
landing.to_csv(path_output + 'landing_impact_sim_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.csv')
dfi.export(landing,
           path_output + 'landing_impact_sim_' +
           income_choice + '_' + outcome_choice + '_' +
           fd_suffix + hhbasis_suffix + '.png',
           fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)
telsendimg(
    conf=tel_config,
    path=path_output + 'landing_impact_sim_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '.png',
    cap='landing_impact_sim_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix
)


# III.B --- Use landing impact to compute dynamic impact using estimated OIRFs from VAR
# Function
def compute_var_dynamic_impact(
        irf,
        list_responses,
        shock,
        shock_size,
        convert_q_to_a
):
    # deep copy of parsed IRF
    dynamic = irf.copy()
    # parse further
    dynamic = dynamic[dynamic['response'].isin(list_responses) & dynamic['shock'].isin([shock])]
    # scale IRFs (originally unit shock)
    dynamic['irf'] = dynamic['irf'] * shock_size
    # reset index
    dynamic = dynamic.reset_index(drop=True)
    # convert from quarterly response to annual response
    if convert_q_to_a:
        dynamic['horizon_year'] = dynamic['horizon'] // 4
        dynamic = dynamic.groupby(['shock', 'response', 'horizon_year'])['irf'].mean().reset_index()
    # output
    return dynamic


# Compute
dynamic = compute_var_dynamic_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_total['Impact on 2023 PC Growth (pp)'].sum(),
    convert_q_to_a=True
)
# Clean up
dynamic = dynamic[dynamic['horizon_year'] < 4]  # ignore final quarter
dynamic = dynamic.rename(columns={'irf': 'dynamic_impact'})
dynamic_rounded = dynamic.copy()
dynamic_rounded['dynamic_impact'] = dynamic_rounded['dynamic_impact'].round(2) 
# Output
dynamic_rounded.to_parquet(path_output + 'dynamic_impact_sim_' +
                   income_choice + '_' + outcome_choice + '_' +
                   fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.parquet')
dynamic_rounded.to_csv(path_output + 'dynamic_impact_sim_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.csv')
dfi.export(dynamic_rounded,
           path_output + 'dynamic_impact_sim_' +
           income_choice + '_' + outcome_choice + '_' +
           fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.png',
           fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)
telsendimg(
    conf=tel_config,
    path=path_output + 'dynamic_impact_sim_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.png',
    cap='dynamic_impact_sim_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

# III.C --- Aggregate impact
# Deep copy
aggregate = dynamic.copy()
# Key in landing impact
aggregate.loc[
    (
            (aggregate['horizon_year'] == 0) & (aggregate['response'] == 'Private Consumption')
    ),
    'landing_impact'
] = landing_total['Impact on 2023 PC Growth (pp)'].sum()
aggregate.loc[
    (
            (aggregate['horizon_year'] == 0) & (aggregate['response'] == 'GDP')
    ),
    'landing_impact'
] = landing_total['Impact on 2023 GDP Growth (pp)'].sum()
aggregate = aggregate.fillna(0)
# Compute total impact across time
aggregate['total_impact'] = aggregate['dynamic_impact'] + aggregate['landing_impact']
# Clean up
aggregate = aggregate[['shock', 'response', 'horizon_year', 'landing_impact', 'dynamic_impact', 'total_impact']]
aggregate_rounded = aggregate.copy()
aggregate_rounded[['landing_impact', 'dynamic_impact', 'total_impact']] = \
    aggregate_rounded[['landing_impact', 'dynamic_impact', 'total_impact']].round(2)
# Output
aggregate_rounded.to_parquet(path_output + 'aggregate_impact_sim_' +
                   income_choice + '_' + outcome_choice + '_' +
                   fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.parquet')
aggregate_rounded.to_csv(path_output + 'aggregate_impact_sim_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.csv')
dfi.export(aggregate_rounded,
           path_output + 'aggregate_impact_sim_' +
           income_choice + '_' + outcome_choice + '_' +
           fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.png',
           fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)
telsendimg(
    conf=tel_config,
    path=path_output + 'aggregate_impact_sim_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.png',
    cap='aggregate_impact_sim_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_impact_sim: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
