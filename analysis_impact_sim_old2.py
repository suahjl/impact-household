# Use income group-specific mpc to simulate total group impact

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    heatmap, pil_img2pdf, wide_grouped_barchart
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
choice_macro_shock = 'Private Consumption'
choice_micro_outcome = 'Consumption'
choices_macro_response = ['GDP', 'Private Consumption', 'Investment', 'CPI', 'NEER', 'MGS 10-Year Yields']
choice_rounds_to_repeat = 1
choice_max_q = 7

# I --- Load MPC, IRF, and disbursement estimates
# IRF estimates
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
# Restrict IRFs to shock of choice
irf = irf[irf['shock'] == choice_macro_shock]
# MPC estimates
mpc = pd.read_parquet(path_output +
                      'params_table_overall_quantile' + '_' +
                      outcome_choice + '_' + income_choice + '_'
                      + fd_suffix + hhbasis_suffix + '.parquet').reset_index()
mpc = mpc[mpc['method'] == 'FE']
mpc = mpc[['quantile', 'Parameter']]
mpc = mpc.rename(columns={'quantile': 'gross_income_group'})
mpc['gross_income_group'] = \
    mpc['gross_income_group'].replace(
        {
            '0-20': 'B20-',
            '20-40': 'B20+',
            '40-60': 'M20-',
            '60-80': 'M20+',
            '80-100': 'T20'
        }
    )
# Disbursement estimates
disb_tiered = pd.read_parquet(path_output + 'cost_matrix_level.parquet')
disb_partial = pd.read_parquet(path_output + 'cost_matrix_partial_level.parquet')
disb_flat = pd.read_parquet(path_output + 'cost_matrix_flat_level.parquet')
disb_b80_tiered = pd.read_parquet(path_output + 'cost_matrix_b80_tiered_level.parquet')
disb_b80_partial = pd.read_parquet(path_output + 'cost_matrix_b80_partial_level.parquet')
disb_b80_flat = pd.read_parquet(path_output + 'cost_matrix_b80_flat_level.parquet')
disb_b80watiered_allotherstiered = \
    pd.read_parquet(path_output + 'cost_matrix_b80watiered_allotherstiered_level.parquet')
disb_b80watiered_allothersflat = \
    pd.read_parquet(path_output + 'cost_matrix_b80watiered_allothersflat_level.parquet')
disb_b80waflat_allothersflat = \
    pd.read_parquet(path_output + 'cost_matrix_b80waflat_allothersflat_level.parquet')

# III.0 --- Set base parameters
list_incgroups = ['B20-', 'B20+', 'M20-', 'M20+', 'T20']
ngdp_2022 = 1788184000000
rgdp_2022 = 1507305000000
npc_2022 = 1029952000000
rpc_2022 = 907573000000
deflator_2022 = 118.65
deflator_2023 = 118.65
pc_deflator_2022 = 113.48
pc_deflator_2023 = 113.48

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


# III.A --- Compute landing impact using estimated MPCs
# Function
def landing_impact_matrix(
        mpc,
        disb,
        incgroups
):
    # Prelims
    mpc = mpc.copy()
    df_disb = disb.copy()
    # Trim total column for convenience
    df_disb = df_disb[incgroups]
    df_landing = df_disb.copy()
    # Compute c = g * mpc
    for incgroup in incgroups:
        df_landing[incgroup] = \
            df_landing[incgroup] \
            * mpc.loc[mpc['gross_income_group'] == incgroup, 'Parameter'].max()
    # Output
    return df_landing


# Compute all impact matrices
landing_tiered = landing_impact_matrix(mpc=mpc, disb=disb_tiered, incgroups=list_incgroups)
landing_partial = landing_impact_matrix(mpc=mpc, disb=disb_partial, incgroups=list_incgroups)
landing_flat = landing_impact_matrix(mpc=mpc, disb=disb_flat, incgroups=list_incgroups)
landing_b80_tiered = landing_impact_matrix(mpc=mpc, disb=disb_b80_tiered, incgroups=list_incgroups)
landing_b80_partial = landing_impact_matrix(mpc=mpc, disb=disb_b80_partial, incgroups=list_incgroups)
landing_b80_flat = landing_impact_matrix(mpc=mpc, disb=disb_b80_flat, incgroups=list_incgroups)
landing_b80watiered_allotherstiered = \
    landing_impact_matrix(mpc=mpc, disb=disb_b80watiered_allotherstiered, incgroups=list_incgroups)
landing_b80watiered_allothersflat = \
    landing_impact_matrix(mpc=mpc, disb=disb_b80watiered_allothersflat, incgroups=list_incgroups)
landing_b80waflat_allothersflat = \
    landing_impact_matrix(mpc=mpc, disb=disb_b80waflat_allothersflat, incgroups=list_incgroups)

# Compute benefit-specific totals (row sums)
landing_tiered['Landing Impact'] = landing_tiered.sum(axis=1)
landing_partial['Landing Impact'] = landing_partial.sum(axis=1)
landing_flat['Landing Impact'] = landing_flat.sum(axis=1)
landing_b80_tiered['Landing Impact'] = landing_b80_tiered.sum(axis=1)
landing_b80_partial['Landing Impact'] = landing_b80_partial.sum(axis=1)
landing_b80_flat['Landing Impact'] = landing_b80_flat.sum(axis=1)
landing_b80watiered_allotherstiered['Landing Impact'] = landing_b80watiered_allotherstiered.sum(axis=1)
landing_b80watiered_allothersflat['Landing Impact'] = landing_b80watiered_allothersflat.sum(axis=1)
landing_b80waflat_allothersflat['Landing Impact'] = landing_b80waflat_allothersflat.sum(axis=1)

# Convert into real PC growth
landing_tiered_rpc = 100 * ((landing_tiered * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_partial_rpc = 100 * ((landing_partial * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_flat_rpc = 100 * ((landing_flat * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80_tiered_rpc = 100 * ((landing_b80_tiered * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80_partial_rpc = 100 * ((landing_b80_partial * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80_flat_rpc = 100 * ((landing_b80_flat * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80watiered_allotherstiered_rpc = \
    100 * ((landing_b80watiered_allotherstiered * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80watiered_allothersflat_rpc = \
    100 * ((landing_b80watiered_allothersflat * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)
landing_b80waflat_allothersflat_rpc = \
    100 * ((landing_b80waflat_allothersflat * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022)

# Convert into real GDP growth
landing_tiered_rgdp = 100 * ((landing_tiered * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_partial_rgdp = 100 * ((landing_partial * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_flat_rgdp = 100 * ((landing_flat * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80_tiered_rgdp = 100 * ((landing_b80_tiered * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80_partial_rgdp = 100 * ((landing_b80_partial * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80_flat_rgdp = 100 * ((landing_b80_flat * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80watiered_allotherstiered_rgdp = \
    100 * ((landing_b80watiered_allotherstiered * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80watiered_allothersflat_rgdp = \
    100 * ((landing_b80watiered_allothersflat * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)
landing_b80waflat_allothersflat_rgdp = \
    100 * ((landing_b80waflat_allothersflat * 1000000000 / (deflator_2023 / 100)) / rgdp_2022)

# Convert landing impact into heat maps
# Nominal
files_landing_nominal = []
fig_landing_tiered = heatmap(
    input=landing_tiered,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'tiered_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): Full Tiering',
    lb=0,
    ub=landing_tiered.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'tiered_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]
fig_landing_partial = heatmap(
    input=landing_partial,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'partial_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): Partial Tiering',
    lb=0,
    ub=landing_partial.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'partial_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]
fig_landing_flat = heatmap(
    input=landing_flat,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'flat_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): Flat Rate',
    lb=0,
    ub=landing_flat.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'flat_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]

fig_landing_b80_tiered = heatmap(
    input=landing_b80_tiered,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_tiered_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 Full Tiering',
    lb=0,
    ub=landing_b80_tiered.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80_tiered_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]
fig_landing_b80_partial = heatmap(
    input=landing_b80_partial,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_partial_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 Partial Tiering',
    lb=0,
    ub=landing_b80_partial.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80_partial_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]
fig_landing_b80_flat = heatmap(
    input=landing_b80_flat,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_flat_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 Flat Rate',
    lb=0,
    ub=landing_b80_flat.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80_flat_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allotherstiered = heatmap(
    input=landing_b80watiered_allotherstiered,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 WA (Tiered), All Others (Tiered)',
    lb=0,
    ub=landing_b80watiered_allotherstiered.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allothersflat = heatmap(
    input=landing_b80watiered_allothersflat,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 WA (Tiered), All Others (Flat)',
    lb=0,
    ub=landing_b80watiered_allothersflat.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]

fig_landing_b80waflat_allothersflat = heatmap(
    input=landing_b80waflat_allothersflat,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (RM bil): B80 WA (Flat), All Others (Flat)',
    lb=0,
    ub=landing_b80waflat_allothersflat.max().max(),
    format='.1f'
)
files_landing_nominal = files_landing_nominal + [path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_' +
                                                 income_choice + '_' + outcome_choice + '_' +
                                                 fd_suffix + hhbasis_suffix]

# PC growth
files_landing_rpc = []
fig_landing_tiered_rpc = heatmap(
    input=landing_tiered_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'tiered_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): Full Tiering',
    lb=0,
    ub=landing_tiered_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'tiered_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]
fig_landing_partial_rpc = heatmap(
    input=landing_partial_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'partial_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): Partial Tiering',
    lb=0,
    ub=landing_partial_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'partial_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]
fig_landing_flat_rpc = heatmap(
    input=landing_flat_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'flat_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): Flat Rate',
    lb=0,
    ub=landing_flat_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'flat_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]

fig_landing_b80_tiered_rpc = heatmap(
    input=landing_b80_tiered_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_tiered_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 Full Tiering',
    lb=0,
    ub=landing_b80_tiered_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80_tiered_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]
fig_landing_b80_partial_rpc = heatmap(
    input=landing_b80_partial_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_partial_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 Partial Tiering',
    lb=0,
    ub=landing_b80_partial_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80_partial_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]
fig_landing_b80_flat_rpc = heatmap(
    input=landing_b80_flat_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_flat_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 Flat Rate',
    lb=0,
    ub=landing_b80_flat_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80_flat_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allotherstiered_rpc = heatmap(
    input=landing_b80watiered_allotherstiered_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 WA (Tiered), All Others (Tiered)',
    lb=0,
    ub=landing_b80watiered_allotherstiered_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allothersflat_rpc = heatmap(
    input=landing_b80watiered_allothersflat_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 WA (Tiered), All Others (Flat)',
    lb=0,
    ub=landing_b80watiered_allothersflat_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]

fig_landing_b80waflat_allothersflat_rpc = heatmap(
    input=landing_b80waflat_allothersflat_rpc,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_rpc_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (PC Growth; pp): B80 WA (Flat), All Others (Flat)',
    lb=0,
    ub=landing_b80waflat_allothersflat_rpc.max().max(),
    format='.1f'
)
files_landing_rpc = files_landing_rpc + [path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_rpc_' +
                                         income_choice + '_' + outcome_choice + '_' +
                                         fd_suffix + hhbasis_suffix]

# GDP growth
files_landing_rgdp = []
fig_landing_tiered_rgdp = heatmap(
    input=landing_tiered_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'tiered_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): Full Tiering',
    lb=0,
    ub=landing_tiered_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'tiered_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]
fig_landing_partial_rgdp = heatmap(
    input=landing_partial_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'partial_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): Partial Tiering',
    lb=0,
    ub=landing_partial_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'partial_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]
fig_landing_flat_rgdp = heatmap(
    input=landing_flat_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'flat_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): Flat Rate',
    lb=0,
    ub=landing_flat_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'flat_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]

fig_landing_b80_tiered_rgdp = heatmap(
    input=landing_b80_tiered_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_tiered_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 Full Tiering',
    lb=0,
    ub=landing_b80_tiered_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80_tiered_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]
fig_landing_b80_partial_rgdp = heatmap(
    input=landing_b80_partial_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_partial_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 Partial Tiering',
    lb=0,
    ub=landing_b80_partial_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80_partial_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]
fig_landing_b80_flat_rgdp = heatmap(
    input=landing_b80_flat_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80_flat_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 Flat Rate',
    lb=0,
    ub=landing_b80_flat_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80_flat_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allotherstiered_rgdp = heatmap(
    input=landing_b80watiered_allotherstiered_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 WA (Tiered), All Others (Tiered)',
    lb=0,
    ub=landing_b80watiered_allotherstiered_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80watiered_allotherstiered_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]

fig_landing_b80watiered_allothersflat_rgdp = heatmap(
    input=landing_b80watiered_allothersflat_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 WA (Tiered), All Others (Flat)',
    lb=0,
    ub=landing_b80watiered_allothersflat_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80watiered_allothersflat_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]

fig_landing_b80waflat_allothersflat_rgdp = heatmap(
    input=landing_b80waflat_allothersflat_rgdp,
    mask=False,
    colourmap='vlag',
    outputfile=path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_rgdp_' +
               income_choice + '_' + outcome_choice + '_' +
               fd_suffix + hhbasis_suffix + '.png',
    title='Annual Landing Impact (GDP Growth; pp): B80 WA (Flat), All Others (Flat)',
    lb=0,
    ub=landing_b80waflat_allothersflat_rgdp.max().max(),
    format='.1f'
)
files_landing_rgdp = files_landing_rgdp + [path_output + 'landing_impact_sim_' + 'b80waflat_allothersflat_rgdp_' +
                                           income_choice + '_' + outcome_choice + '_' +
                                           fd_suffix + hhbasis_suffix]

# Compile landing impact heatmaps
pil_img2pdf(
    list_images=files_landing_nominal,
    extension='png',
    pdf_name=path_output + 'landing_impact_sim_' + 'nominal_' +
             income_choice + '_' + outcome_choice + '_' +
             fd_suffix + hhbasis_suffix
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'landing_impact_sim_' + 'nominal_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '.pdf',
    cap='landing_impact_sim_' + 'nominal_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix
)

pil_img2pdf(
    list_images=files_landing_rpc,
    extension='png',
    pdf_name=path_output + 'landing_impact_sim_' + 'rpc_' +
             income_choice + '_' + outcome_choice + '_' +
             fd_suffix + hhbasis_suffix
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'landing_impact_sim_' + 'rpc_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '.pdf',
    cap='landing_impact_sim_' + 'nominal_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix
)

pil_img2pdf(
    list_images=files_landing_rgdp,
    extension='png',
    pdf_name=path_output + 'landing_impact_sim_' + 'rgdp_' +
             income_choice + '_' + outcome_choice + '_' +
             fd_suffix + hhbasis_suffix
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'landing_impact_sim_' + 'rgdp_' +
         income_choice + '_' + outcome_choice + '_' +
         fd_suffix + hhbasis_suffix + '.pdf',
    cap='landing_impact_sim_' + 'nominal_' +
        income_choice + '_' + outcome_choice + '_' +
        fd_suffix + hhbasis_suffix
)


# III.B --- Use landing impact to compute indirect impact using estimated OIRFs from VAR
# Function
def compute_var_indirect_impact(
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
    indirect = indirect.rename(columns={'irf': 'indirect_impact'})
    indirect_rounded = indirect.copy()
    indirect_rounded['indirect_impact'] = indirect_rounded['indirect_impact'].round(2)
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


def export_dfi_parquet_csv(input, file_name):
    input.to_parquet(file_name + '.parquet')
    input.to_csv(file_name + '.csv')
    dfi.export(input, file_name + '.png',
               fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)


# Compute
indirect_tiered, indirect_tiered_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_tiered_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_partial, indirect_partial_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_partial_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_flat, indirect_flat_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_flat_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)

indirect_b80_tiered, indirect_b80_tiered_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80_tiered_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_b80_partial, indirect_b80_partial_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80_partial_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_b80_flat, indirect_b80_flat_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80_flat_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)

indirect_b80watiered_allotherstiered, indirect_b80watiered_allotherstiered_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80watiered_allotherstiered_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_b80watiered_allothersflat, indirect_b80watiered_allothersflat_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80watiered_allothersflat_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)
indirect_b80waflat_allothersflat, indirect_b80waflat_allothersflat_rounded = compute_var_indirect_impact(
    irf=irf,
    list_responses=choices_macro_response,
    shock=choice_macro_shock,
    shock_size=landing_b80waflat_allothersflat_rpc.loc['All Benefits', 'Landing Impact'].max(),
    convert_q_to_a=True,
    max_q=choice_max_q
)

# Output
export_dfi_parquet_csv_telegram(
    input=indirect_tiered_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_partial_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_flat_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=indirect_b80_tiered_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80_tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_b80_partial_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80_partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_b80_flat_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80_flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=indirect_b80watiered_allotherstiered_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80watiered_allotherstiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_b80watiered_allothersflat_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80watiered_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=indirect_b80waflat_allothersflat_rounded,
    file_name=path_output + 'indirect_impact_sim_' + 'b80waflat_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.C --- Aggregate impact
# Functions
def create_aggregate_impact(indirect, landing_rpc, landing_rgdp):
    # Deep copy
    aggregate = indirect.copy()
    # Input landing impact (real PC and real GDP growth pp)
    aggregate.loc[
        (
                (aggregate['horizon_year'] == 0) & (aggregate['response'] == 'Private Consumption')
        ),
        'landing_impact'
    ] = landing_rpc.loc['All Benefits', 'Landing Impact'].max()
    aggregate.loc[
        (
                (aggregate['horizon_year'] == 0) & (aggregate['response'] == 'GDP')
        ),
        'landing_impact'
    ] = landing_rgdp.loc['All Benefits', 'Landing Impact'].max()
    # Compute total impact across time
    aggregate['landing_impact'] = aggregate['landing_impact'].fillna(0)
    aggregate['total_impact'] = aggregate['indirect_impact'] + aggregate['landing_impact']
    # Clean up
    aggregate = aggregate[
        ['shock', 'response', 'horizon_year', 'landing_impact', 'indirect_impact', 'total_impact']
    ]
    aggregate_rounded = aggregate.copy()
    aggregate_rounded[['landing_impact', 'indirect_impact', 'total_impact']] = \
        aggregate_rounded[['landing_impact', 'indirect_impact', 'total_impact']].round(2)
    # Output
    return aggregate, aggregate_rounded


# Compute
aggregate_tiered, aggregate_tiered_rounded = create_aggregate_impact(
    indirect=indirect_tiered,
    landing_rpc=landing_tiered_rpc,
    landing_rgdp=landing_tiered_rgdp
)
aggregate_partial, aggregate_partial_rounded = create_aggregate_impact(
    indirect=indirect_partial,
    landing_rpc=landing_partial_rpc,
    landing_rgdp=landing_partial_rgdp
)
aggregate_flat, aggregate_flat_rounded = create_aggregate_impact(
    indirect=indirect_flat,
    landing_rpc=landing_flat_rpc,
    landing_rgdp=landing_flat_rgdp
)

aggregate_b80_tiered, aggregate_b80_tiered_rounded = create_aggregate_impact(
    indirect=indirect_b80_tiered,
    landing_rpc=landing_b80_tiered_rpc,
    landing_rgdp=landing_b80_tiered_rgdp
)
aggregate_b80_partial, aggregate_b80_partial_rounded = create_aggregate_impact(
    indirect=indirect_b80_partial,
    landing_rpc=landing_b80_partial_rpc,
    landing_rgdp=landing_b80_partial_rgdp
)
aggregate_b80_flat, aggregate_b80_flat_rounded = create_aggregate_impact(
    indirect=indirect_b80_flat,
    landing_rpc=landing_b80_flat_rpc,
    landing_rgdp=landing_b80_flat_rgdp
)

aggregate_b80watiered_allotherstiered, aggregate_b80watiered_allotherstiered_rounded = create_aggregate_impact(
    indirect=indirect_b80watiered_allotherstiered,
    landing_rpc=landing_b80watiered_allotherstiered_rpc,
    landing_rgdp=landing_b80watiered_allotherstiered_rgdp
)
aggregate_b80watiered_allothersflat, aggregate_b80watiered_allothersflat_rounded = create_aggregate_impact(
    indirect=indirect_b80watiered_allothersflat,
    landing_rpc=landing_b80watiered_allothersflat_rpc,
    landing_rgdp=landing_b80watiered_allothersflat_rgdp
)
aggregate_b80waflat_allothersflat, aggregate_b80waflat_allothersflat_rounded = create_aggregate_impact(
    indirect=indirect_b80waflat_allothersflat,
    landing_rpc=landing_b80waflat_allothersflat_rpc,
    landing_rgdp=landing_b80waflat_allothersflat_rgdp
)

# Export
export_dfi_parquet_csv_telegram(
    input=aggregate_tiered_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_partial_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_flat_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=aggregate_b80_tiered_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80_tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_b80_partial_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80_partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_b80_flat_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80_flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=aggregate_b80watiered_allotherstiered_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80watiered_allotherstiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_b80watiered_allothersflat_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80watiered_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=aggregate_b80waflat_allothersflat_rounded,
    file_name=path_output + 'aggregate_impact_sim_' + 'b80waflat_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.D --- Repeated years aggregate impact
# Function
def repeated_aggregate_impact(aggregate, landing_rpc, landing_rgdp, indirect, rounds_to_repeat):
    # Deep copy
    repeated_agg = aggregate.copy()
    # Repeat landing impact beyond year 0 for (PC and GDP)
    repeated_agg.loc[
        (
                (repeated_agg['horizon_year'] <= rounds_to_repeat) &
                (repeated_agg['shock'] == 'Private Consumption') &
                (repeated_agg['response'] == 'Private Consumption')
        ),
        'landing_impact'] = landing_rpc.loc['All Benefits', 'Landing Impact'].max()
    repeated_agg.loc[
        (
                (repeated_agg['horizon_year'] <= rounds_to_repeat) &
                (repeated_agg['shock'] == 'Private Consumption') &
                (repeated_agg['response'] == 'GDP')
        ),
        'landing_impact'] = landing_rgdp.loc['All Benefits', 'Landing Impact'].max()
    # Repeat indirect impact beyond year 0
    round = 1
    while round <= rounds_to_repeat:
        repeated_agg.loc[repeated_agg['horizon_year'] >= round, 'indirect_impact'] = \
            repeated_agg['indirect_impact'] + \
            indirect.groupby(['shock', 'response'])['indirect_impact'].shift(round)
        round += 1
    # Recalculate total impact
    repeated_agg['total_impact'] = repeated_agg['landing_impact'] + repeated_agg['indirect_impact']
    # Clean up
    repeated_agg_rounded = repeated_agg.copy()
    repeated_agg_rounded[['landing_impact', 'indirect_impact', 'total_impact']] = \
        repeated_agg_rounded[['landing_impact', 'indirect_impact', 'total_impact']].round(2)
    # Output
    return repeated_agg, repeated_agg_rounded


# Compute
repeated_agg_tiered, repeated_agg_tiered_rounded = repeated_aggregate_impact(
    aggregate=aggregate_tiered,
    landing_rpc=landing_tiered_rpc,
    landing_rgdp=landing_tiered_rgdp,
    indirect=indirect_tiered,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_partial, repeated_agg_partial_rounded = repeated_aggregate_impact(
    aggregate=aggregate_partial,
    landing_rpc=landing_partial_rpc,
    landing_rgdp=landing_partial_rgdp,
    indirect=indirect_partial,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_flat, repeated_agg_flat_rounded = repeated_aggregate_impact(
    aggregate=aggregate_flat,
    landing_rpc=landing_flat_rpc,
    landing_rgdp=landing_flat_rgdp,
    indirect=indirect_flat,
    rounds_to_repeat=choice_rounds_to_repeat
)

repeated_agg_b80_tiered, repeated_agg_b80_tiered_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80_tiered,
    landing_rpc=landing_b80_tiered_rpc,
    landing_rgdp=landing_b80_tiered_rgdp,
    indirect=indirect_b80_tiered,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_b80_partial, repeated_agg_b80_partial_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80_partial,
    landing_rpc=landing_b80_partial_rpc,
    landing_rgdp=landing_b80_partial_rgdp,
    indirect=indirect_b80_partial,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_b80_flat, repeated_agg_b80_flat_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80_flat,
    landing_rpc=landing_b80_flat_rpc,
    landing_rgdp=landing_b80_flat_rgdp,
    indirect=indirect_b80_flat,
    rounds_to_repeat=choice_rounds_to_repeat
)

repeated_agg_b80watiered_allotherstiered, repeated_agg_b80watiered_allotherstiered_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80watiered_allotherstiered,
    landing_rpc=landing_b80watiered_allotherstiered_rpc,
    landing_rgdp=landing_b80watiered_allotherstiered_rgdp,
    indirect=indirect_b80watiered_allotherstiered,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_b80watiered_allothersflat, repeated_agg_b80watiered_allothersflat_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80watiered_allothersflat,
    landing_rpc=landing_b80watiered_allothersflat_rpc,
    landing_rgdp=landing_b80watiered_allothersflat_rgdp,
    indirect=indirect_b80watiered_allothersflat,
    rounds_to_repeat=choice_rounds_to_repeat
)
repeated_agg_b80waflat_allothersflat, repeated_agg_b80waflat_allothersflat_rounded = repeated_aggregate_impact(
    aggregate=aggregate_b80waflat_allothersflat,
    landing_rpc=landing_b80waflat_allothersflat_rpc,
    landing_rgdp=landing_b80waflat_allothersflat_rgdp,
    indirect=indirect_b80waflat_allothersflat,
    rounds_to_repeat=choice_rounds_to_repeat
)

# Export
export_dfi_parquet_csv_telegram(
    input=repeated_agg_tiered_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_partial_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_flat_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80_tiered_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80_tiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80_partial_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80_partial_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80_flat_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80_flat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)

export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80watiered_allotherstiered_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80watiered_allotherstiered_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80watiered_allothersflat_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80watiered_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)
export_dfi_parquet_csv_telegram(
    input=repeated_agg_b80waflat_allothersflat_rounded,
    file_name=path_output + 'repeated_agg_impact_sim_' + 'b80waflat_allothersflat_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.E --- Compile repeated aggregate impact of all combos
def compile_all_combos(
        flat, partial, tiered,
        b80_flat, b80_partial, b80_tiered,
        b80watiered_allotherstiered, b80watiered_allothersflat, b80waflat_allothersflat
):
    # Deep copies
    df_tiered = tiered.copy()
    df_partial = partial.copy()
    df_flat = flat.copy()

    df_b80_tiered = b80_tiered.copy()
    df_b80_partial = b80_partial.copy()
    df_b80_flat = b80_flat.copy()

    df_b80watiered_allotherstiered = b80watiered_allotherstiered.copy()
    df_b80watiered_allothersflat = b80watiered_allothersflat.copy()
    df_b80waflat_allothersflat = b80waflat_allothersflat.copy()

    # Change columns
    df_tiered = df_tiered.rename(
        columns={
            'landing_impact': 'Full Tiering: Landing',
            'indirect_impact': 'Full Tiering: Indirect',
            'total_impact': 'Full Tiering: Total'
        }
    )
    df_partial = df_partial.rename(
        columns={
            'landing_impact': 'Partial Tiering: Landing',
            'indirect_impact': 'Partial Tiering: Indirect',
            'total_impact': 'Partial Tiering: Total'
        }
    )
    df_flat = df_flat.rename(
        columns={
            'landing_impact': 'Flat Rate: Landing',
            'indirect_impact': 'Flat Rate: Indirect',
            'total_impact': 'Flat Rate: Total'
        }
    )

    df_b80_tiered = df_b80_tiered.rename(
        columns={
            'landing_impact': 'B80 Full Tiering: Landing',
            'indirect_impact': 'B80 Full Tiering: Indirect',
            'total_impact': 'B80 Full Tiering: Total'
        }
    )
    df_b80_partial = df_b80_partial.rename(
        columns={
            'landing_impact': 'B80 Partial Tiering: Landing',
            'indirect_impact': 'B80 Partial Tiering: Indirect',
            'total_impact': 'B80 Partial Tiering: Total'
        }
    )
    df_b80_flat = df_b80_flat.rename(
        columns={
            'landing_impact': 'B80 Flat Rate: Landing',
            'indirect_impact': 'B80 Flat Rate: Indirect',
            'total_impact': 'B80 Flat Rate: Total'
        }
    )

    df_b80watiered_allotherstiered = df_b80watiered_allotherstiered.rename(
        columns={
            'landing_impact': 'B80 WA (Tiered), All Others (Tiered): Landing',
            'indirect_impact': 'B80 WA (Tiered), All Others (Tiered): Indirect',
            'total_impact': 'B80 WA (Tiered), All Others (Tiered): Total'
        }
    )
    df_b80watiered_allothersflat = df_b80watiered_allothersflat.rename(
        columns={
            'landing_impact': 'B80 WA (Tiered), All Others (Flat): Landing',
            'indirect_impact': 'B80 WA (Tiered), All Others (Flat): Indirect',
            'total_impact': 'B80 WA (Tiered), All Others (Flat): Total'
        }
    )
    df_b80waflat_allothersflat = df_b80waflat_allothersflat.rename(
        columns={
            'landing_impact': 'B80 WA (Flat), All Others (Flat): Landing',
            'indirect_impact': 'B80 WA (Flat), All Others (Flat): Indirect',
            'total_impact': 'B80 WA (Flat), All Others (Flat): Total'
        }
    )

    # Combine
    allcombos = df_tiered.merge(df_partial, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_flat, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80_tiered, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80_partial, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80_flat, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80watiered_allotherstiered, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80watiered_allothersflat, how='left', on=['shock', 'response', 'horizon_year'])
    allcombos = allcombos.merge(df_b80waflat_allothersflat, how='left', on=['shock', 'response', 'horizon_year'])

    # Clean up
    allcombos_rounded = allcombos.copy()
    list_impact_cols = [
        'Full Tiering: Landing', 'Full Tiering: Indirect', 'Full Tiering: Total',
        'Partial Tiering: Landing', 'Partial Tiering: Indirect', 'Partial Tiering: Total',
        'Flat Rate: Landing', 'Flat Rate: Indirect', 'Flat Rate: Total',
        'B80 Full Tiering: Landing', 'B80 Full Tiering: Indirect', 'B80 Full Tiering: Total',
        'B80 Partial Tiering: Landing', 'B80 Partial Tiering: Indirect', 'B80 Partial Tiering: Total',
        'B80 Flat Rate: Landing', 'B80 Flat Rate: Indirect', 'B80 Flat Rate: Total',
        'B80 WA (Tiered), All Others (Tiered): Landing',
        'B80 WA (Tiered), All Others (Tiered): Indirect',
        'B80 WA (Tiered), All Others (Tiered): Total',
        'B80 WA (Tiered), All Others (Flat): Landing',
        'B80 WA (Tiered), All Others (Flat): Indirect',
        'B80 WA (Tiered), All Others (Flat): Total',
        'B80 WA (Flat), All Others (Flat): Landing',
        'B80 WA (Flat), All Others (Flat): Indirect',
        'B80 WA (Flat), All Others (Flat): Total'
    ]
    allcombos_rounded[list_impact_cols] = allcombos_rounded[list_impact_cols].round(2)

    # Output
    return allcombos, allcombos_rounded


allcombos, allcombos_rounded = \
    compile_all_combos(
        flat=repeated_agg_flat,
        partial=repeated_agg_partial,
        tiered=repeated_agg_tiered,
        b80_tiered=repeated_agg_b80_tiered,
        b80_partial=repeated_agg_b80_partial,
        b80_flat=repeated_agg_b80_flat,
        b80watiered_allotherstiered=repeated_agg_b80watiered_allotherstiered,
        b80watiered_allothersflat=repeated_agg_b80watiered_allothersflat,
        b80waflat_allothersflat=repeated_agg_b80waflat_allothersflat
    )
export_dfi_parquet_csv(
    input=allcombos_rounded,
    file_name=path_output + 'allcombos_' +
              income_choice + '_' + outcome_choice + '_' +
              fd_suffix + hhbasis_suffix + '_' + macro_suffix
)


# III.F --- Compile repeated aggregate impact of all combos, but split by response variable
def split_allcombos_heatmap_telegram(allcombos, list_shocks, list_responses, total_only, dict_ltavg):
    # Parameters
    if total_only:
        total_only_suffix = 'totalonly_'
        list_impact_cols = [
            'Full Tiering: Total',
            'Partial Tiering: Total',
            'Flat Rate: Total',
            'B80 Full Tiering: Total',
            'B80 Partial Tiering: Total', 'B80 Flat Rate: Total',
            'B80 WA (Tiered), All Others (Tiered): Total',
            'B80 WA (Tiered), All Others (Flat): Total',
            'B80 WA (Flat), All Others (Flat): Total'
        ]
    elif not total_only:
        total_only_suffix = ''
        list_impact_cols = [
            'Full Tiering: Landing', 'Full Tiering: Indirect', 'Full Tiering: Total',
            'Partial Tiering: Landing', 'Partial Tiering: Indirect', 'Partial Tiering: Total',
            'Flat Rate: Landing', 'Flat Rate: Indirect', 'Flat Rate: Total',
            'B80 Full Tiering: Landing', 'B80 Full Tiering: Indirect', 'B80 Full Tiering: Total',
            'B80 Partial Tiering: Landing', 'B80 Partial Tiering: Indirect',
            'B80 Partial Tiering: Total',
            'B80 Flat Rate: Landing', 'B80 Flat Rate: Indirect', 'B80 Flat Rate: Total',
            'B80 WA (Tiered), All Others (Tiered): Landing',
            'B80 WA (Tiered), All Others (Tiered): Indirect',
            'B80 WA (Tiered), All Others (Tiered): Total',
            'B80 WA (Tiered), All Others (Flat): Landing',
            'B80 WA (Tiered), All Others (Flat): Indirect',
            'B80 WA (Tiered), All Others (Flat): Total',
            'B80 WA (Flat), All Others (Flat): Landing',
            'B80 WA (Flat), All Others (Flat): Indirect',
            'B80 WA (Flat), All Others (Flat): Total'
        ]
    # Deep copies
    allcombos_full = allcombos.copy()
    # Beautify time horizon
    allcombos_full['horizon_year'] = allcombos_full['horizon_year'] + 1
    allcombos_full['horizon_year'] = 'Year ' + allcombos_full['horizon_year'].astype('str')
    # Split into shock-response specific heatmaps
    list_files = []
    for shock in list_shocks:
        for response in tqdm(list_responses):
            # A. Growth impact
            allcombos_sub = allcombos_full[
                (allcombos_full['shock'] == shock) &
                (allcombos_full['response'] == response)
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
                outputfile=path_output + 'allcombos_' + total_only_suffix + shock + '_' + response + '_' +
                           income_choice + '_' + outcome_choice + '_' +
                           fd_suffix + hhbasis_suffix + '.png',
                title='Breakdown of Impact on ' + response + ' (pp)',
                lb=0,
                ub=allcombos_sub.max().max(),
                format='.1f'
            )
            list_files = list_files + [path_output + 'allcombos_' + total_only_suffix + shock + '_' + response + '_' +
                                       income_choice + '_' + outcome_choice + '_' +
                                       fd_suffix + hhbasis_suffix]
            # Generate barchart
            bar_allcombos_sub = wide_grouped_barchart(
                data=allcombos_sub,
                y_cols=list_impact_cols,
                group_col='horizon_year',
                main_title='Breakdown of Impact on ' + response + ' (pp)',
                decimal_points=1,
                group_colours=['lightblue', 'lightpink']
            )
            allcombos_sub = allcombos_sub.reset_index()  # so that horizon_year is part of the data frame, not index
            bar_allcombos_sub.write_image(
                path_output + 'bar_allcombos_' + total_only_suffix + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix + '.png')
            list_files = list_files + [
                path_output + 'bar_allcombos_' + total_only_suffix + shock + '_' + response + '_' +
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
                path_output + 'bar_allcombos_sub_level_gap_' + total_only_suffix + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix + '.png')
            list_files = list_files + [
                path_output + 'bar_allcombos_sub_level_gap_' + total_only_suffix + shock + '_' + response + '_' +
                income_choice + '_' + outcome_choice + '_' +
                fd_suffix + hhbasis_suffix]

    # X. Compile PDF
    pil_img2pdf(list_images=list_files,
                extension='png',
                pdf_name=path_output + 'allcombos_' + total_only_suffix + 'shock_response_' +
                         income_choice + '_' + outcome_choice + '_' +
                         fd_suffix + hhbasis_suffix)
    # X. Send telegram
    telsendfiles(
        conf=tel_config,
        path=path_output + 'allcombos_' + total_only_suffix + 'shock_response_' +
             income_choice + '_' + outcome_choice + '_' +
             fd_suffix + hhbasis_suffix + '.pdf',
        cap='allcombos_' + total_only_suffix + 'shock_response_' +
            income_choice + '_' + outcome_choice + '_' +
            fd_suffix + hhbasis_suffix
    )


split_allcombos_heatmap_telegram(
    allcombos=allcombos,
    list_shocks=[choice_macro_shock],
    list_responses=choices_macro_response,
    total_only=False,
    dict_ltavg=dict_ltavg
)

# III.G --- Compile repeated aggregate impact of all combos, but split by response variable; total impact only
split_allcombos_heatmap_telegram(
    allcombos=allcombos,
    list_shocks=[choice_macro_shock],
    list_responses=choices_macro_response,
    total_only=True,
    dict_ltavg=dict_ltavg
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_impact_sim: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
