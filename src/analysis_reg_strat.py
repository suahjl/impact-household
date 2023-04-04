# Regression analysis, but stratified

import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols, heatmap
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

# I --- Load data
df = pd.read_parquet(path_data + 'hies_consol_agg_balanced_mean.parquet')
df_ind = pd.read_parquet(path_data + 'hies_consol_ind.parquet')

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
        'adolescent_group',
        'child_group',
        'male',
        'birth_year_group',
        'marriage',
        'emp_status',
        'industry',
        'occupation'
    ]
df[col_groups] = df[col_groups].astype('str')
df['cohort_code'] = df[col_groups].sum(axis=1)
df = df.drop(col_groups, axis=1)

# logs
col_cons = ['cons_01', 'cons_02', 'cons_03', 'cons_04',
            'cons_05', 'cons_06', 'cons_07', 'cons_08',
            'cons_09', 'cons_10', 'cons_11', 'cons_12',
            'cons_13',
            'cons_01_12', 'cons_01_13']
col_inc = ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']
for i in col_cons + col_inc:
    pass
    # df[i] = np.log(df[i])
    # df_ind[i] = np.log(df_ind[i])

# First diff
if use_first_diff:
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
        'cons_13': 'Financial Expenses'
    }
# Define outcome lists
list_outcome_choices = ['cons_01_13', 'cons_01_12'] + \
                       ['cons_0' + str(i) for i in range(1, 10)] + \
                       ['cons_1' + str(i) for i in range(0, 4)]

# Estimates
round = 1
for outcome in tqdm(list_outcome_choices):
    mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
        fe_reg(
            df=df,
            y_col=outcome,
            x_cols=[income_choice],
            i_col='cohort_code',
            t_col='year',
            fixed_effects=True,
            time_effects=False,
            cov_choice='robust'
        )
    params_table_fe['outcome_variable'] = outcome
    if round == 1:
        params_table_fe_consol = pd.DataFrame(params_table_fe.loc[income_choice]).transpose()
    elif round >= 1:
        params_table_fe_consol = pd.concat(
            [params_table_fe_consol, pd.DataFrame(params_table_fe.loc[income_choice]).transpose()],
            axis=0)
    round += 1
params_table_fe_consol['outcome_variable'] = \
    params_table_fe_consol['outcome_variable'] \
        .replace(dict_cons_nice)
params_table_fe_consol = params_table_fe_consol.set_index('outcome_variable')
params_table_fe_consol = params_table_fe_consol.astype('float')
heatmap_params_table_fe_consol = heatmap(
    input=params_table_fe_consol,
    mask=False,
    colourmap='vlag',
    outputfile='output/params_table_fe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
    title='',
    lb=0,
    ub=0.6,
    format='.3f'
)
telsendimg(conf=tel_config,
           path='output/params_table_fe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
           cap='params_table_fe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix)

round = 1
for outcome in tqdm(list_outcome_choices):
    mod_timefe, res_timefe, params_table_timefe, joint_teststats_timefe, reg_det_timefe = \
        fe_reg(
            df=df,
            y_col=outcome,
            x_cols=[income_choice],
            i_col='cohort_code',
            t_col='year',
            fixed_effects=False,
            time_effects=True,
            cov_choice='robust'
        )
    params_table_timefe['outcome_variable'] = outcome
    if round == 1:
        params_table_timefe_consol = pd.DataFrame(params_table_timefe.loc[income_choice]).transpose()
    elif round >= 1:
        params_table_timefe_consol = pd.concat(
            [params_table_timefe_consol, pd.DataFrame(params_table_timefe.loc[income_choice]).transpose()],
            axis=0)
    round += 1
params_table_timefe_consol['outcome_variable'] = \
    params_table_timefe_consol['outcome_variable'] \
        .replace(dict_cons_nice)
params_table_timefe_consol = params_table_timefe_consol.set_index('outcome_variable')
params_table_timefe_consol = params_table_timefe_consol.astype('float')
heatmap_params_table_timefe_consol = heatmap(
    input=params_table_timefe_consol,
    mask=False,
    colourmap='vlag',
    outputfile='output/params_table_timefe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
    title='',
    lb=0,
    ub=0.6,
    format='.3f'
)
telsendimg(conf=tel_config,
           path='output/params_table_timefe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
           cap='params_table_timefe_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix)

round = 1
for outcome in tqdm(list_outcome_choices):
    mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = \
        re_reg(
            df=df,
            y_col=outcome,
            x_cols=[income_choice],
            i_col='cohort_code',
            t_col='year',
            cov_choice='robust'
        )
    params_table_re['outcome_variable'] = outcome
    if round == 1:
        params_table_re_consol = pd.DataFrame(params_table_re.loc[income_choice]).transpose()
    elif round >= 1:
        params_table_re_consol = pd.concat(
            [params_table_re_consol, pd.DataFrame(params_table_re.loc[income_choice]).transpose()],
            axis=0)
    round += 1
params_table_re_consol['outcome_variable'] = \
    params_table_re_consol['outcome_variable'] \
        .replace(dict_cons_nice)
params_table_re_consol = params_table_re_consol.set_index('outcome_variable')
params_table_re_consol = params_table_re_consol.astype('float')
heatmap_params_table_re_consol = heatmap(
    input=params_table_re_consol,
    mask=False,
    colourmap='vlag',
    outputfile='output/params_table_re_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
    title='',
    lb=0,
    ub=0.6,
    format='.3f'
)
telsendimg(conf=tel_config,
           path='output/params_table_re_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
           cap='params_table_re_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix)

round = 1
for outcome in tqdm(list_outcome_choices):
    mod_ind_ols, res_ind_ols, params_table_ind_ols, joint_teststats_ind_ols, reg_det_ind_ols = \
        reg_ols(
            df=df_ind,
            eqn=outcome + ' ~ ' + income_choice + ' + ' +
                'C(state) + urban + C(education) + C(ethnicity) + ' +
                'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
                'C(child_group) + male + C(birth_year_group) + C(marriage) + ' +
                'C(emp_status) + C(industry) + C(occupation) + C(year)'
        )
    params_table_ind_ols['outcome_variable'] = outcome
    if round == 1:
        params_table_ind_ols_consol = pd.DataFrame(params_table_ind_ols.loc[income_choice]).transpose()
    elif round >= 1:
        params_table_ind_ols_consol = pd.concat(
            [params_table_ind_ols_consol, pd.DataFrame(params_table_ind_ols.loc[income_choice]).transpose()],
            axis=0)
    round += 1
params_table_ind_ols_consol['outcome_variable'] = \
    params_table_ind_ols_consol['outcome_variable'] \
        .replace(dict_cons_nice)
params_table_ind_ols_consol = params_table_ind_ols_consol.set_index('outcome_variable')
params_table_ind_ols_consol = params_table_ind_ols_consol.astype('float')
heatmap_params_table_ind_ols_consol = heatmap(
    input=params_table_ind_ols_consol,
    mask=False,
    colourmap='vlag',
    outputfile='output/params_table_ind_ols_consol_strat_cons' + '_' + income_choice + '_' + fd_suffix + '.png',
    title='',
    lb=0,
    ub=0.6,
    format='.3f'
)
telsendimg(conf=tel_config,
           path='output/params_table_ind_ols_consol_strat_cons' + '_' + income_choice + '.png',
           cap='params_table_ind_ols_consol_strat_cons' + '_' + income_choice)


# III.B --- Estimation: stratify by income groups (individual levels; no FD option)
def gen_gross_income_group(data, aggregation):
    if aggregation == 1:
        data.loc[(data['gross_income'] >= data['gross_income'].quantile(q=0.8)), 'gross_income_group'] = 4
        data.loc[((data['gross_income'] >= data['gross_income'].quantile(q=0.6)) &
                  (data['gross_income'] < data['gross_income'].quantile(q=0.8))), 'gross_income_group'] = 3
        data.loc[((data['gross_income'] >= data['gross_income'].quantile(q=0.4)) &
                  (data['gross_income'] < data['gross_income'].quantile(q=0.6))), 'gross_income_group'] = 2
        data.loc[((data['gross_income'] >= data['gross_income'].quantile(q=0.2)) &
                  (data['gross_income'] < data['gross_income'].quantile(q=0.4))), 'gross_income_group'] = 1
        data.loc[(data['gross_income'] < data['gross_income'].quantile(q=0.2)), 'gross_income_group'] = 0
    elif aggregation == 2:
        data.loc[(data['gross_income'] >= data['gross_income'].quantile(q=0.8)), 'gross_income_group'] = 2
        data.loc[((data['gross_income'] >= data['gross_income'].quantile(q=0.4)) &
                  (data['gross_income'] < data['gross_income'].quantile(q=0.8))), 'gross_income_group'] = 1
        data.loc[(data['gross_income'] < data['gross_income'].quantile(q=0.4)), 'gross_income_group'] = 0


gen_gross_income_group(data=df_ind, aggregation=1)
round = 1
for income_group in tqdm(range(0, int(df_ind['gross_income_group'].max() + 1))):
    d = df_ind[df_ind['gross_income_group'] == income_group]
    mod_ind_ols, res_ind_ols, params_table_ind_ols, joint_teststats_ind_ols, reg_det_ind_ols = \
        reg_ols(
            df=d,
            eqn=outcome_choice + ' ~ ' + income_choice + ' + ' +
                'C(state) + urban + C(education) + C(ethnicity) + ' +
                'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
                'C(child_group) + male + C(birth_year_group) + C(marriage) + ' +
                'C(emp_status) + C(industry) + C(occupation) + C(year)'
        )
    params_table_ind_ols['outcome_variable'] = outcome_choice
    params_table_ind_ols['gross_income_group'] = income_group
    if round == 1:
        params_table_ind_ols_consol = pd.DataFrame(params_table_ind_ols.loc[income_choice]).transpose()
    elif round >= 1:
        params_table_ind_ols_consol = pd.concat(
            [params_table_ind_ols_consol, pd.DataFrame(params_table_ind_ols.loc[income_choice]).transpose()],
            axis=0)
    round += 1
params_table_ind_ols_consol['outcome_variable'] = \
    params_table_ind_ols_consol['outcome_variable']\
        .replace(dict_cons_nice)
params_table_ind_ols_consol = params_table_ind_ols_consol.set_index('outcome_variable')
params_table_ind_ols_consol = params_table_ind_ols_consol[['gross_income_group', 'Parameter', 'LowerCI', 'UpperCI']]
dfi.export(params_table_ind_ols_consol,
           'output/params_table_ind_ols_consol_strat_incomeq' + '_' + outcome_choice + '_' + income_choice + '.png')
telsendimg(conf=tel_config,
           path='output/params_table_ind_ols_consol_strat_incomeq' + '_' + outcome_choice + '_' + income_choice + '.png',
           cap='params_table_ind_ols_consol_strat_incomeq' + '_' + outcome_choice + '_' + income_choice)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_reg_strat: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
