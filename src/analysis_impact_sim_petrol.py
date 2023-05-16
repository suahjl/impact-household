# Impact of fuel price subsidy rationalisation

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    heatmap, pil_img2pdf, barchart, wide_grouped_barchart
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

dict_scenarios_annual_shocks = {
    'Immediate': 0.564,
    '3 Months': 0.520,
    '6 Months': 0.453
}
list_scenarios = list(dict_scenarios_annual_shocks.keys())

pc_ltavg = 6.934
gfcf_ltavg = 2.450
gdp_ltavg = 4.951
neer_ltavg = -2.436
cpi_ltavg = 1.925
mgs10y_ltavg = -0.08
dict_ltavg = {
    'GDP': gdp_ltavg,
    'Private Consumption': pc_ltavg,
    'Investment': gfcf_ltavg,
    'CPI': cpi_ltavg,
    'NEER': neer_ltavg,
    'MGS 10-Year Yields': mgs10y_ltavg
}

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
scenario_count = 1
for scenario, shock_size in tqdm(dict_scenarios_annual_shocks.items()):
    indirect, indirect_rounded = compute_var_impact(
        irf=irf,
        list_responses=choices_macro_response,
        shock=choice_macro_shock,
        shock_size=shock_size,
        convert_q_to_a=True,
        max_q=choice_max_q
    )
    indirect = indirect.rename(columns={'impact': scenario})
    indirect_rounded = indirect.rename(columns={'impact': scenario})
    if scenario_count == 1:
        indirect_consol = indirect.copy()
        indirect_rounded_consol = indirect_rounded.copy()
    elif scenario_count > 1:
        indirect_consol = indirect_consol.merge(
            indirect,
            on=['shock', 'response', 'horizon_year']
        )
        indirect_rounded_consol = indirect_rounded_consol.merge(
            indirect_rounded,
            on=['shock', 'response', 'horizon_year']
        )
    scenario_count += 1

# Visualisation
# Dataframe image
export_dfi_parquet_csv_telegram(
    input=indirect_rounded_consol,
    file_name=path_output + 'impact_sim_petrol_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.B --- Bar chart of impact by responses
def compile_barcharts_telegram(indirect, dict_ltavg, list_scenarios):
    indirect['horizon_year'] = 'Year ' + (indirect['horizon_year'] + 1).astype('str')  # convert to presentable form
    list_files = []
    for shock in [choice_macro_shock]:
        for response in tqdm(choices_macro_response):
            # Parameters
            list_impact_cols = list_scenarios
            # A. Growth impact
            allcombos_sub = indirect[
                (indirect['shock'] == shock) &
                (indirect['response'] == response)
                ].copy()
            # Keep only time horizon + impact estimates
            for i in ['shock', 'response']:
                del allcombos_sub[i]
            allcombos_sub = allcombos_sub.set_index('horizon_year')
            allcombos_sub = allcombos_sub[list_impact_cols]
            # Generate heatmaps
            fig_allcombos_sub = heatmap(
                input=allcombos_sub,
                mask=False,
                colourmap='vlag',
                outputfile=path_output + 'impact_sim_petrol_' + shock + '_' + response + '_' +
                           income_choice + '_' + outcome_choice + '_' +
                           fd_suffix + hhbasis_suffix + '.png',
                title='Breakdown of Impact on ' + response + ' (pp)',
                lb=0,
                ub=allcombos_sub.max().max(),
                format='.1f'
            )
            list_files = list_files + [path_output + 'impact_sim_petrol_' + shock + '_' + response + '_' +
                                       income_choice + '_' + outcome_choice + '_' +
                                       fd_suffix + hhbasis_suffix]
            # Generate bar chart
            allcombos_sub = allcombos_sub.reset_index()  # so that horizon_year is part of the data frame, not index
            bar_allcombos_sub = wide_grouped_barchart(
                data=allcombos_sub,
                y_cols=list_impact_cols,
                group_col='horizon_year',
                main_title='Breakdown of Impact on ' + response + ' (pp)',
                decimal_points=1,
                group_colours=['lightblue', 'lightpink']
            )
            bar_allcombos_sub.write_image(
                path_output + 'bar_impact_sim_petrol_' + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix + '.png'
            )
            list_files = list_files + [path_output + 'bar_impact_sim_petrol_' + shock + '_' + response + '_' +
                                       income_choice + '_' + outcome_choice + '_' +
                                       fd_suffix + hhbasis_suffix]

            # B. Levels impact
            # Create data frame to house levels
            allcombos_sub_cf = allcombos_sub.copy()
            allcombos_sub_level = allcombos_sub.copy()
            # Compute counterfactual levels using LT avg
            ltavg = dict_ltavg[response]
            for horizon in range(0, len(allcombos_sub_cf)):
                allcombos_sub_cf.loc[allcombos_sub_cf['horizon_year'] == 'Year ' + str(horizon + 1),
                                     list_impact_cols] = \
                    100 * (1 + ltavg / 100) ** (horizon + 1)
            # Compute realised levels
            round_sub_levels = 1
            for horizon in range(0, len(allcombos_sub_cf)):
                if round_sub_levels == 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level['horizon_year'] == 'Year ' + str(horizon + 1),
                        list_impact_cols] = \
                        100 * (1 + (ltavg + allcombos_sub.loc[
                            allcombos_sub['horizon_year'] == 'Year ' + str(horizon + 1),
                            list_impact_cols]) / 100)
                elif round_sub_levels > 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level['horizon_year'] == 'Year ' + str(horizon + 1),
                        list_impact_cols] = \
                        allcombos_sub_level[list_impact_cols].shift(1) * (1 + (ltavg + allcombos_sub.loc[
                            allcombos_sub['horizon_year'] == 'Year ' + str(horizon + 1),
                            list_impact_cols]) / 100)
                round_sub_levels += 1
            # Compute levels impact
            allcombos_sub_level_gap = allcombos_sub_level.copy()
            allcombos_sub_level_gap[list_impact_cols] = \
                allcombos_sub_level_gap[list_impact_cols] - allcombos_sub_cf[list_impact_cols]
            # Create bar chart
            bar_allcombos_sub_level_gap = wide_grouped_barchart(
                data=allcombos_sub_level_gap,
                y_cols=list_impact_cols,
                group_col='horizon_year',
                main_title='Implied Levels Impact (Assuming LT Avg) on ' + response + ' (% Baseline)',
                decimal_points=1,
                group_colours=['lightblue', 'lightpink']
            )
            bar_allcombos_sub_level_gap.write_image(
                path_output + 'bar_impact_sim_petrol_level_gap_' + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix + '.png')
            list_files = list_files + [
                path_output + 'bar_impact_sim_petrol_level_gap_' + shock + '_' + response + '_' +
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


compile_barcharts_telegram(
    indirect=indirect_consol,
    dict_ltavg=dict_ltavg,
    list_scenarios=list_scenarios
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_impact_sim_petrol: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
