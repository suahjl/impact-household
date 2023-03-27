# Regression analysis, but stratified

import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols
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

# I --- Load data
df = pd.read_parquet(path_data + 'hies_consol_agg_balanced.parquet')
df_ind = pd.read_parquet(path_data + 'hies_consol_ind.parquet')

# II --- Pre-analysis prep
# Redefine year
for i, j in zip([2014, 2016, 2019], [1, 2, 3]):
    df.loc[df['year'] == i, 'year'] = j
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

# III.A --- Estimation: stratify by outcomes (consumption categories)
# Define outcome lists
list_outcome_choices = ['cons_01_13', 'cons_01_12'] + \
                       ['cons_0' + str(i) for i in range(1, 10)] + \
                       ['cons_1' + str(i) for i in range(0, 4)]

# Estimates
round = 1
for outcome_choice in tqdm(list_outcome_choices):
    mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
        fe_reg(
            df=df,
            y_col=outcome_choice,
            x_cols=['gross_income'],
            i_col='cohort_code',
            t_col='year',
            fixed_effects=True,
            time_effects=False,
            cov_choice='robust'
        )
    params_table_fe['outcome_variable'] = outcome_choice
    if round == 1:
        params_table_fe_consol = pd.DataFrame(params_table_fe.loc['gross_income']).transpose()
    elif round >= 1:
        params_table_fe_consol = pd.concat(
            [params_table_fe_consol, pd.DataFrame(params_table_fe.loc['gross_income']).transpose()],
            axis=0)
    round += 1
params_table_fe_consol = params_table_fe_consol.set_index('outcome_variable')
dfi.export(params_table_fe_consol, 'output/params_table_fe_consol.png')
telsendimg(conf=tel_config,
           path='output/params_table_fe_consol.png',
           cap='params_table_fe_consol')

round = 1
for outcome_choice in tqdm(list_outcome_choices):
    mod_timefe, res_timefe, params_table_timefe, joint_teststats_timefe, reg_det_timefe = \
        fe_reg(
            df=df,
            y_col=outcome_choice,
            x_cols=['gross_income'],
            i_col='cohort_code',
            t_col='year',
            fixed_effects=False,
            time_effects=True,
            cov_choice='robust'
        )
    params_table_timefe['outcome_variable'] = outcome_choice
    if round == 1:
        params_table_timefe_consol = pd.DataFrame(params_table_timefe.loc['gross_income']).transpose()
    elif round >= 1:
        params_table_timefe_consol = pd.concat(
            [params_table_timefe_consol, pd.DataFrame(params_table_timefe.loc['gross_income']).transpose()],
            axis=0)
    round += 1
params_table_timefe_consol = params_table_timefe_consol.set_index('outcome_variable')
dfi.export(params_table_timefe_consol, 'output/params_table_timefe_consol.png')
telsendimg(conf=tel_config,
           path='output/params_table_timefe_consol.png',
           cap='params_table_timefe_consol')

round = 1
for outcome_choice in tqdm(list_outcome_choices):
    mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = \
        re_reg(
            df=df,
            y_col=outcome_choice,
            x_cols=['gross_income'],
            i_col='cohort_code',
            t_col='year',
            cov_choice='robust'
        )
    params_table_re['outcome_variable'] = outcome_choice
    if round == 1:
        params_table_re_consol = pd.DataFrame(params_table_re.loc['gross_income']).transpose()
    elif round >= 1:
        params_table_re_consol = pd.concat(
            [params_table_re_consol, pd.DataFrame(params_table_re.loc['gross_income']).transpose()],
            axis=0)
    round += 1
params_table_re_consol = params_table_re_consol.set_index('outcome_variable')
dfi.export(params_table_re_consol, 'output/params_table_re_consol.png')
telsendimg(conf=tel_config,
           path='output/params_table_re_consol.png',
           cap='params_table_re_consol')

round = 1
for outcome_choice in tqdm(list_outcome_choices):
    mod_ind_ols, res_ind_ols, params_table_ind_ols, joint_teststats_ind_ols, reg_det_ind_ols = \
        reg_ols(
            df=df_ind,
            eqn=outcome_choice + ' ~ gross_income + ' +
                'C(state) + urban + C(education) + C(ethnicity) + ' +
                'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
                'C(child_group) + male + C(birth_year_group) + C(marriage) + ' +
                'C(emp_status) + C(industry) + C(occupation) + C(year)'
        )
    params_table_ind_ols['outcome_variable'] = outcome_choice
    if round == 1:
        params_table_ind_ols_consol = pd.DataFrame(params_table_ind_ols.loc['gross_income']).transpose()
    elif round >= 1:
        params_table_ind_ols_consol = pd.concat(
            [params_table_ind_ols_consol, pd.DataFrame(params_table_ind_ols.loc['gross_income']).transpose()],
            axis=0)
    round += 1
params_table_ind_ols_consol = params_table_ind_ols_consol.set_index('outcome_variable')
dfi.export(params_table_ind_ols_consol, 'output/params_table_ind_ols_consol.png')
telsendimg(conf=tel_config,
           path='output/params_table_ind_ols_consol.png',
           cap='params_table_ind_ols_consol')


# III.B --- Estimation: stratify by income groups
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

gen_gross_income_group(data=df_ind, aggregation=2)
round = 1
for income_group in tqdm(range(0, int(df_ind['gross_income_group'].max() + 1))):
    d = df_ind[df_ind['gross_income_group'] == income_group]
    mod_ind_ols, res_ind_ols, params_table_ind_ols, joint_teststats_ind_ols, reg_det_ind_ols = \
        reg_ols(
            df=d,
            eqn='cons_01_12' + ' ~ gross_income + ' +
                'C(state) + urban + C(education) + C(ethnicity) + ' +
                'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
                'C(child_group) + male + C(birth_year_group) + C(marriage) + ' +
                'C(emp_status) + C(industry) + C(occupation) + C(year)'
        )
    params_table_ind_ols['outcome_variable'] = 'cons_01_12'
    params_table_ind_ols['income_group'] = income_group
    if round == 1:
        params_table_ind_ols_consol = pd.DataFrame(params_table_ind_ols.loc['gross_income']).transpose()
    elif round >= 1:
        params_table_ind_ols_consol = pd.concat(
            [params_table_ind_ols_consol, pd.DataFrame(params_table_ind_ols.loc['gross_income']).transpose()],
            axis=0)
    round += 1
params_table_ind_ols_consol = params_table_ind_ols_consol.set_index('outcome_variable')
dfi.export(params_table_ind_ols_consol, 'output/params_table_ind_ols_consol_byincomeq.png')
telsendimg(conf=tel_config,
           path='output/params_table_ind_ols_consol_byincomeq.png',
           cap='params_table_ind_ols_consol_byincomeq')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_reg: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
