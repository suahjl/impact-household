# Net impact of all policy options

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    heatmap, heatmap_layered, pil_img2pdf, wide_grouped_barchart
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

# Which impact
cumul_net_impact_level_gap = ast.literal_eval(os.getenv('CUMUL_NET_IMPACT_LEVELS_GAP'))
if cumul_net_impact_level_gap:
    level_suffix = 'level_gap_'
    level_chart_title = ' (Levels)'
elif not cumul_net_impact_level_gap:
    level_suffix = ''
    level_chart_title = ''

# Which parameters
choices_macro_response = ['GDP', 'Private Consumption', 'Investment', 'CPI', 'NEER', 'MGS 10-Year Yields']
# Long-term averages to scale ranges
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
# I --- Load cash transfers impact, fuel subsidy rationalisation impact, and elec subsidy rationalisation impact
# Load data
cash = pd.read_parquet(path_output + 'allcombos_' + level_suffix + 'consol_' + 'shock_response_' +
                       income_choice + '_' + outcome_choice + '_' +
                       fd_suffix + hhbasis_suffix + '.parquet')
cash = cash[['response', 'horizon_year'] + [i for i in cash.columns if (': Total' in i)]]
cash.columns = [i.replace(': Total', '') for i in cash.columns]
scenarios_cash = [i for i in cash.columns if ('response' not in i) and ('horizon_year' not in i)]
# scenarios_cash = [i for i in cash.columns if
#                   ('response' not in i) and
#                   ('horizon_year' not in i) and
#                   ('Tiered' not in i)]
# scenarios_cash = [i for i in cash.columns if
#                   ('response' not in i) and
#                   ('horizon_year' not in i) and
#                   ('Flat' not in i)]

petrol = pd.read_parquet(path_output + 'impact_sim_petrol_' + level_suffix +
                         income_choice + '_' + outcome_choice + '_' +
                         fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.parquet')
del petrol['shock']

# elec = pd.read_parquet(path_output + 'impact_sim_elec_' + level_suffix +
#                        income_choice + '_' + outcome_choice + '_' +
#                        fd_suffix + hhbasis_suffix + '_' + macro_suffix + '.parquet')
# del elec['shock']
# Change horizon_year
cash['horizon_year'] = 'Year ' + (cash['horizon_year'] + 1).astype('str')
petrol['horizon_year'] = 'Year ' + (petrol['horizon_year'] + 1).astype('str')
# elec['horizon_year'] = 'Year ' + (elec['horizon_year'] + 1).astype('str')

# II --- Generate elec + petrol frame
# Set index to avoid miscalc
petrol = petrol.set_index(['response', 'horizon_year'])
# elec = elec.set_index(['response', 'horizon_year'])
# Create combined policy frame
elec_petrol = petrol.copy()
# elec_petrol.columns = [i + ' + Elec Sub Removal' for i in elec_petrol.columns]
# for i in list(elec_petrol.columns):
#     elec_petrol[i] = elec_petrol[i] + elec['Full Removal']
# Reset index
elec_petrol = elec_petrol.reset_index()
# elec = elec.reset_index()
petrol = petrol.reset_index()

# III --- Generate cash + (elec + petrol) frame
# Set index to avoid miscalc
elec_petrol = elec_petrol.set_index(['response', 'horizon_year'])
cash = cash.set_index(['response', 'horizon_year'])
# Iterate through elec_petrol scenarios over cash scenarios
net_base = cash.copy()
net_base['Petrol Scenarios'] = ''
col = net_base.pop('Petrol Scenarios')
net_base.insert(0, col.name, col)
round_y_scenario = 0
for y_scenario in list(elec_petrol.columns):
    # Deep copy
    net_suby = net_base.copy()
    # Combine impact of cash scenarios + Yth elec & petrol scenario
    for cash_scenario in scenarios_cash:
        net_suby[cash_scenario] = net_suby[cash_scenario] + elec_petrol[y_scenario]
    net_suby['Petrol Scenarios'] = y_scenario
    # Consolidate
    if round_y_scenario == 0:
        net = net_suby.copy()
    elif round_y_scenario > 0:
        net = pd.concat([net, net_suby], axis=0)  # top-down
    # Increase tracker
    round_y_scenario += 1
# Reset indices
elec_petrol = elec_petrol.reset_index()
cash = cash.reset_index()
net = net.reset_index()
# Move petrol_ & elec scenarios to first column in net
col = net.pop('Petrol Scenarios')
net.insert(0, col.name, col)
# Sort
net = net.sort_values(by=['response', 'horizon_year', 'Petrol Scenarios'], ascending=[True, True, True])

# IV --- Generate cumulative net impact (same as CIRFs)
net_cumul = net.groupby(['Petrol Scenarios', 'response'])[scenarios_cash].sum().reset_index()
net_cumul = net_cumul.sort_values(by=['response', 'Petrol Scenarios'], ascending=[True, True])

# V.A --- Generate year-by-year, and variable-by-variable net impact frames (full)
list_file_names = []
for response, ltavg in zip(choices_macro_response, list(dict_ltavg.values())):
    for horizon in list(net['horizon_year'].unique()):
        # Subset + deep copy
        net_sub = net.loc[(net['response'] == response) & (net['horizon_year'] == horizon),
                          ['Petrol Scenarios'] + scenarios_cash].copy()
        # Set index
        net_sub = net_sub.set_index('Petrol Scenarios')
        # Generate heatmap
        file_name = path_output + 'net_impact_' + level_suffix + response + '_' + horizon + '_' + \
                    income_choice + '_' + outcome_choice + '_' + \
                    fd_suffix + hhbasis_suffix
        fig = heatmap(
            input=net_sub,
            mask=False,
            colourmap='vlag',
            outputfile=file_name + '.png',
            title='Net Impact on ' + response + ' For ' + horizon + level_chart_title,
            lb=-1 * np.abs(ltavg),
            ub=np.abs(ltavg),
            format='.2f'
        )
        list_file_names = list_file_names + [file_name]
# Consolidate pdf
file_pdf = path_output + 'net_impact_' + level_suffix + \
           income_choice + '_' + outcome_choice + '_' + \
           fd_suffix + hhbasis_suffix
pil_img2pdf(
    list_images=list_file_names,
    extension='png',
    pdf_name=file_pdf
)
telsendfiles(
    conf=tel_config,
    path=file_pdf + '.pdf',
    cap=file_pdf
)

# V.B --- Generate year-by-year, and variable-by-variable net impact frames (compressed)
list_file_names = []
for response, ltavg in zip(choices_macro_response, list(dict_ltavg.values())):
    for horizon in list(net['horizon_year'].unique()):
        # Subset + deep copy
        net_sub = net.loc[(net['response'] == response) & (net['horizon_year'] == horizon),
                          ['Petrol Scenarios'] + scenarios_cash].copy()
        # Set index
        net_sub = net_sub.set_index('Petrol Scenarios')
        # COMPRESS Y AXIS (PETROL SCENARIOS)
        net_sub_val = pd.DataFrame(net_sub.mean(axis=0)).transpose()
        net_sub_val.index = pd.Series(['Immediate to 6 Months'])
        net_sub_val.index.name = 'Petrol Scenarios'
        # GENERATE DISPLAY FRAME
        net_sub_disp = pd.DataFrame(net_sub.min(axis=0)).round(2).transpose().astype('str') + \
                       '\n to \n' + \
                       pd.DataFrame(net_sub.max(axis=0)).round(2).transpose().astype('str')
        # Generate heatmap
        file_name = path_output + 'net_impact_compressed_' + level_suffix + response + '_' + horizon + '_' + \
                    income_choice + '_' + outcome_choice + '_' + \
                    fd_suffix + hhbasis_suffix
        fig = heatmap_layered(
            actual_input=net_sub_val,
            disp_input=net_sub_disp,
            mask=False,
            colourmap='vlag',
            outputfile=file_name + '.png',
            title='Net Impact on ' + response + ' For ' + horizon + level_chart_title,
            lb=-1 * np.abs(ltavg),
            ub=np.abs(ltavg),
            format='s'
        )
        list_file_names = list_file_names + [file_name]
# Consolidate pdf
file_pdf = path_output + 'net_impact_compressed_' + level_suffix + \
           income_choice + '_' + outcome_choice + '_' + \
           fd_suffix + hhbasis_suffix
pil_img2pdf(
    list_images=list_file_names,
    extension='png',
    pdf_name=file_pdf
)
telsendfiles(
    conf=tel_config,
    path=file_pdf + '.pdf',
    cap=file_pdf
)

# VI.A --- Generate variable-by-variable but cumulative net impact (full)
list_file_names = []
for response, ltavg in zip(choices_macro_response, list(dict_ltavg.values())):
    # Subset + deep copy
    net_sub = net_cumul.loc[net_cumul['response'] == response,
                            ['Petrol Scenarios'] + scenarios_cash].copy()
    # Set index
    net_sub = net_sub.set_index('Petrol Scenarios')
    # Generate heatmap
    file_name = path_output + 'net_impact_cumul_' + level_suffix + response + '_' + \
                income_choice + '_' + outcome_choice + '_' + \
                fd_suffix + hhbasis_suffix
    fig = heatmap(
        input=net_sub,
        mask=False,
        colourmap='vlag',
        outputfile=file_name + '.png',
        title='Cumulative Net Impact on ' + response + level_chart_title,
        lb=-1 * np.abs(ltavg) * 2,
        ub=np.abs(ltavg) * 2,
        format='.2f'
    )
    list_file_names = list_file_names + [file_name]
# Consolidate pdf
file_pdf = path_output + 'net_impact_cumul_' + level_suffix + \
           income_choice + '_' + outcome_choice + '_' + \
           fd_suffix + hhbasis_suffix
pil_img2pdf(
    list_images=list_file_names,
    extension='png',
    pdf_name=file_pdf
)
telsendfiles(
    conf=tel_config,
    path=file_pdf + '.pdf',
    cap=file_pdf
)

# VI.B --- Generate variable-by-variable but cumulative net impact (compressed)
list_file_names = []
for response, ltavg in zip(choices_macro_response, list(dict_ltavg.values())):
    # Subset + deep copy
    net_sub = net_cumul.loc[net_cumul['response'] == response,
                            ['Petrol Scenarios'] + scenarios_cash].copy()
    # Set index
    net_sub = net_sub.set_index('Petrol Scenarios')
    # COMPRESS Y AXIS (PETROL SCENARIOS)
    net_sub_val = pd.DataFrame(net_sub.mean(axis=0)).transpose()
    net_sub_val.index = pd.Series(['Immediate to 6 Months'])
    net_sub_val.index.name = 'Petrol Scenarios'
    # GENERATE DISPLAY FRAME
    net_sub_disp = pd.DataFrame(net_sub.min(axis=0)).round(2).transpose().astype('str') + \
                   '\n to \n' + \
                   pd.DataFrame(net_sub.max(axis=0)).round(2).transpose().astype('str')
    # Generate heatmap
    file_name = path_output + 'net_impact_cumul_compressed_' + level_suffix + response + '_' + \
                income_choice + '_' + outcome_choice + '_' + \
                fd_suffix + hhbasis_suffix
    fig = heatmap_layered(
        actual_input=net_sub_val,
        disp_input=net_sub_disp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name + '.png',
        title='Cumulative Net Impact on ' + response + level_chart_title,
        lb=-1 * np.abs(ltavg) * 2,
        ub=np.abs(ltavg) * 2,
        format='s'
    )
    list_file_names = list_file_names + [file_name]
# Consolidate pdf
file_pdf = path_output + 'net_impact_cumul_compressed_' + level_suffix + \
           income_choice + '_' + outcome_choice + '_' + \
           fd_suffix + hhbasis_suffix
pil_img2pdf(
    list_images=list_file_names,
    extension='png',
    pdf_name=file_pdf
)
telsendfiles(
    conf=tel_config,
    path=file_pdf + '.pdf',
    cap=file_pdf
)

# VI.B --- Generate variable-by-variable but cumulative net impact (squiggly line)
list_file_names = []
for response, ltavg in zip(choices_macro_response, list(dict_ltavg.values())):
    # Subset + deep copy
    net_sub = net_cumul.loc[net_cumul['response'] == response,
                            ['Petrol Scenarios'] + scenarios_cash].copy()
    # Set index
    net_sub = net_sub.set_index('Petrol Scenarios')
    # COMPRESS Y AXIS (PETROL SCENARIOS)
    net_sub_val = pd.DataFrame(net_sub.mean(axis=0)).transpose()
    net_sub_val.index = pd.Series(['Immediate to 6 Months'])
    net_sub_val.index.name = 'Petrol Scenarios'
    # GENERATE DISPLAY FRAME
    net_sub_disp = pd.DataFrame(net_sub.mean(axis=0)).round(1).transpose().astype('str')
    # Generate heatmap
    file_name = path_output + 'net_impact_cumul_squiggly_' + level_suffix + response + '_' + \
                income_choice + '_' + outcome_choice + '_' + \
                fd_suffix + hhbasis_suffix
    fig = heatmap_layered(
        actual_input=net_sub_val,
        disp_input=net_sub_disp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name + '.png',
        title='Cumulative Net Impact on ' + response + level_chart_title,
        lb=-1 * np.abs(ltavg) * 2,
        ub=np.abs(ltavg) * 2,
        format='s'
    )
    list_file_names = list_file_names + [file_name]
# Consolidate pdf
file_pdf = path_output + 'net_impact_cumul_squiggly_' + level_suffix + \
           income_choice + '_' + outcome_choice + '_' + \
           fd_suffix + hhbasis_suffix
pil_img2pdf(
    list_images=list_file_names,
    extension='png',
    pdf_name=file_pdf
)
telsendfiles(
    conf=tel_config,
    path=file_pdf + '.pdf',
    cap=file_pdf
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_impact_sim: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
