# Impact of fuel price subsidy rationalisation

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    heatmap, pil_img2pdf, barchart
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
hhbasis_cohorts_with_hhsize = ast.literal_eval(os.getenv('HHBASIS_COHORTS_WITH_HHSIZE'))

# Which parameters
choice_macro_shock = 'RON95'
choices_macro_response = ['GDP', 'Private Consumption', 'Investment', 'CPI', 'NEER', 'MGS 10-Year Yields']
choice_fuel_price_revision = 1
choice_max_q = 7

# I --- Load MPC, IRF, and disbursement estimates
# IRF estimates (check which IRF is used)
irf = pd.read_parquet(path_output + 'macro_var_petrol_irf_all_varyx_narrative_avg_' + macro_suffix + '.parquet')
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
    'maxgepu': 'Uncertainty Shocks',
    'ron95': 'RON95'
}
for i in ['shock', 'response']:
    irf[i] = irf[i].replace(dict_cols_nice)
# Restrict IRFs to shock of choice
irf = irf[irf['shock'] == choice_macro_shock]


# III.0 --- Set base parameters

# III.A --- Use landing impact to compute indirect impact using estimated OIRFs from VAR
# Function
def compute_var_impact(
        irf,
        list_responses,
        shock,
        shock_size,
        convert_q_to_a,
        max_q
):
    # deep copy of parsed IRF
    indirect = irf.copy()
    # parse further
    indirect = indirect[indirect['response'].isin(list_responses) & indirect['shock'].isin([shock])]
    # scale IRFs (originally unit shock)
    indirect['irf'] = indirect['irf'] * shock_size
    # limit quarters
    indirect = indirect[indirect['horizon'] <= max_q]
    # reset index
    indirect = indirect.reset_index(drop=True)
    # convert from quarterly response to annual response
    if convert_q_to_a:
        indirect['horizon_year'] = indirect['horizon'] // 4
        indirect = indirect.groupby(['shock', 'response', 'horizon_year'])['irf'].mean().reset_index()
        # Clean up
        indirect = indirect[indirect['horizon_year'] < 4]  # ignore final quarter
    # Clean up
    indirect = indirect.rename(columns={'irf': 'impact'})
    indirect_rounded = indirect.copy()
    indirect_rounded['impact'] = indirect_rounded['impact'].round(2)
    # output
    return indirect, indirect_rounded


def export_dfi_parquet_csv_telegram(input, file_name):
    input.to_parquet(file_name + '.parquet')
    input.to_csv(file_name + '.csv')
    dfi.export(input, file_name + '.png',
               fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)
    telsendimg(
        conf=tel_config,
        path=file_name + '.png',
        cap=file_name
    )


# Compute
indirect, indirect_rounded = compute_var_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=1,
    convert_q_to_a=True,
    max_q=choice_max_q
)

# Visualisation
# Dataframe image
export_dfi_parquet_csv_telegram(
    input=indirect_rounded,
    file_name=path_output + 'impact_sim_petrol_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.B --- Bar chart of impact by responses
def compile_barcharts_telegram(indirect):
    indirect['horizon_year'] = 'Year ' + (indirect['horizon_year'] + 1).astype('str')  # convert to presentable form
    list_files = []
    for shock in [choice_macro_shock]:
        for response in choices_macro_response:
            # Generate bar chart
            bar_allcombos_sub = barchart(
                data=indirect[indirect['response'] == response],
                y_col='impact',
                x_col='horizon_year',
                main_title='Breakdown of Impact on ' + response + ' (pp)',
                decimal_points=1
            )
            bar_allcombos_sub.write_image(
                path_output + 'bar_impact_sim_petrol_' + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix + '.png'
            )
            list_files = list_files + [path_output + 'bar_impact_sim_petrol_' + shock + '_' + response + '_' +
                                       income_choice + '_' + outcome_choice + '_' +
                                       fd_suffix + hhbasis_suffix]
    # Compile PDF
    pil_img2pdf(list_images=list_files,
                extension='png',
                pdf_name=path_output + 'bar_impact_sim_petrol_' + 'shock_response_' +
                         income_choice + '_' + outcome_choice + '_' +
                         fd_suffix + hhbasis_suffix)
    # Send telegram
    telsendfiles(
        conf=tel_config,
        path=path_output + 'bar_impact_sim_petrol_' + 'shock_response_' +
             income_choice + '_' + outcome_choice + '_' +
             fd_suffix + hhbasis_suffix + '.pdf',
        cap='bar_impact_sim_petrol_' + 'shock_response_' +
            income_choice + '_' + outcome_choice + '_' +
            fd_suffix + hhbasis_suffix
    )


compile_barcharts_telegram(indirect=indirect)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_impact_sim_petrol: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
