# Regression analysis, but stratified, and for subgroups

import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols, heatmap, pil_img2pdf, barchart
from tabulate import tabulate
from tqdm import tqdm
import dataframe_image as dfi
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/hies_consol/'
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

# --------- Analysis Starts (only cohort pseudo-panel data regressions) ---------

def load_clean_estimate(
        input_suffix,
        opt_income,
        opt_first_diff,
        opt_show_ci
):
    # I --- Load data
    df = pd.read_parquet(path_data + 'hies_consol_agg_balanced_' + input_suffix + hhbasis_suffix + '.parquet')

    # II --- Pre-analysis prep
    # Redefine year
    df = df.rename(columns={'_time': 'year'})
    # Keep only entity + time + time-variant variables
    col_groups = \
        [
            'state',
            'urban',
            'education',
            'ethnicity',
            'malaysian',
            'income_gen_members_group',
            'working_adult_females_group',
            'non_working_adult_females_group',
        'working_age_group',
            'adolescent_group',
            'child_group',
            'elderly_group',
            'male',
            'birth_year_group',
            'marriage',
            'emp_status',
            'industry',
            'occupation'
        ]
    if hhbasis_adj_analysis and hhbasis_cohorts_with_hhsize:
        col_groups = col_groups + ['hh_size_group']
    df[col_groups] = df[col_groups].astype('str')
    df['cohort_code'] = df[col_groups].sum(axis=1)
    df = df.drop(col_groups, axis=1)

    # logs
    col_cons = ['cons_01', 'cons_02', 'cons_03', 'cons_04',
                'cons_05', 'cons_06', 'cons_07', 'cons_08',
                'cons_09', 'cons_10', 'cons_11', 'cons_12',
                'cons_13',
                'cons_01_12', 'cons_01_13',
                'cons_0722_fuel', 'cons_07_ex_bigticket']
    col_inc = ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']
    for i in col_cons + col_inc:
        pass
        # df[i] = np.log(df[i])
        # df_ind[i] = np.log(df_ind[i])

    # First diff
    if opt_first_diff:
        for y in col_cons + col_inc:
            # df[y] = 100 * (df[y] - df.groupby('cohort_code')[y].shift(1)) / df.groupby('cohort_code')[y].shift(1)
            df[y] = 100 * (df[y] - df.groupby('cohort_code')[y].shift(1))
        df = df.dropna(axis=0)

    # III.A --- Estimation: stratify by outcomes (consumption categories)
    # Nice names for consumption categories
    dict_cons_nice = \
        {
            'cons_01_12': 'Consumption',
            'cons_01_13': 'Consumption + Fin. Expenses',
            'cons_01': 'Food & Beverages',
            'cons_02': 'Alcohol & Tobacco',
            'cons_03': 'Clothing & Footwear',
            'cons_04': 'Rent & Utilities',
            'cons_05': 'Furnishing, HH Equipment & Maintenance',
            'cons_06': 'Healthcare',
            'cons_07': 'Transport',
            'cons_08': 'Communication',
            'cons_09': 'Recreation & Culture',
            'cons_10': 'Education',
            'cons_11': 'Restaurant & Hotels',
            'cons_12': 'Misc',
            'cons_13': 'Financial Expenses',
            'cons_0722_fuel': 'Transport: Fuel Only',
            'cons_07_ex_bigticket': 'Transport ex. Vehicles & Maintenance'
        }
    # Define outcome lists
    list_outcome_choices = ['cons_01_13', 'cons_01_12'] + \
                           ['cons_0' + str(i) for i in range(1, 10)] + \
                           ['cons_1' + str(i) for i in range(0, 4)] + \
                           ['cons_0722_fuel', 'cons_07_ex_bigticket']

    # Estimates: FE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
            fe_reg(
                df=df,
                y_col=outcome,
                x_cols=[opt_income],
                i_col='cohort_code',
                t_col='year',
                fixed_effects=True,
                time_effects=False,
                cov_choice='robust'
            )
        params_table_fe['outcome_variable'] = outcome
        if round == 1:
            params_table_fe_consol = pd.DataFrame(params_table_fe.loc[opt_income]).transpose()
        elif round >= 1:
            params_table_fe_consol = pd.concat(
                [params_table_fe_consol, pd.DataFrame(params_table_fe.loc[opt_income]).transpose()],
                axis=0)
        round += 1
    params_table_fe_consol['outcome_variable'] = \
        params_table_fe_consol['outcome_variable'] \
            .replace(dict_cons_nice)
    params_table_fe_consol = params_table_fe_consol.set_index('outcome_variable')
    params_table_fe_consol = params_table_fe_consol.astype('float')
    if not opt_show_ci:
        for col in ['LowerCI', 'UpperCI']:
            del params_table_fe_consol[col]

    # Estimates: time FE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        mod_timefe, res_timefe, params_table_timefe, joint_teststats_timefe, reg_det_timefe = \
            fe_reg(
                df=df,
                y_col=outcome,
                x_cols=[opt_income],
                i_col='cohort_code',
                t_col='year',
                fixed_effects=False,
                time_effects=True,
                cov_choice='robust'
            )
        params_table_timefe['outcome_variable'] = outcome
        if round == 1:
            params_table_timefe_consol = pd.DataFrame(params_table_timefe.loc[opt_income]).transpose()
        elif round >= 1:
            params_table_timefe_consol = pd.concat(
                [params_table_timefe_consol, pd.DataFrame(params_table_timefe.loc[opt_income]).transpose()],
                axis=0)
        round += 1
    params_table_timefe_consol['outcome_variable'] = \
        params_table_timefe_consol['outcome_variable'] \
            .replace(dict_cons_nice)
    params_table_timefe_consol = params_table_timefe_consol.set_index('outcome_variable')
    params_table_timefe_consol = params_table_timefe_consol.astype('float')
    if not opt_show_ci:
        for col in ['LowerCI', 'UpperCI']:
            del params_table_timefe_consol[col]

    # Estimates: RE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = \
            re_reg(
                df=df,
                y_col=outcome,
                x_cols=[opt_income],
                i_col='cohort_code',
                t_col='year',
                cov_choice='robust'
            )
        params_table_re['outcome_variable'] = outcome
        if round == 1:
            params_table_re_consol = pd.DataFrame(params_table_re.loc[opt_income]).transpose()
        elif round >= 1:
            params_table_re_consol = pd.concat(
                [params_table_re_consol, pd.DataFrame(params_table_re.loc[opt_income]).transpose()],
                axis=0)
        round += 1
    params_table_re_consol['outcome_variable'] = \
        params_table_re_consol['outcome_variable'] \
            .replace(dict_cons_nice)
    params_table_re_consol = params_table_re_consol.set_index('outcome_variable')
    params_table_re_consol = params_table_re_consol.astype('float')
    if not opt_show_ci:
        for col in ['LowerCI', 'UpperCI']:
            del params_table_re_consol[col]

    # IV --- Output
    return params_table_fe_consol, params_table_timefe_consol, params_table_re_consol

# Loop to estimate all scenarios
dict_subgroups = {
    '': ''
}
list_filenames_fe = []
list_filenames_timefe = []
list_filenames_re = []

# --------- Analysis Ends ---------

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_reg_strat_subgroups: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')