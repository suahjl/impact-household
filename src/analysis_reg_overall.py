# Regression analysis, but overall

import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols
from tabulate import tabulate
import dataframe_image as dfi
from tqdm import tqdm
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

# III --- Estimation
# Setup
# Execute cohort-level

mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
    fe_reg(
        df=df,
        y_col=outcome_choice,
        x_cols=[income_choice],
        i_col='cohort_code',
        t_col='year',
        fixed_effects=True,
        time_effects=False,
        cov_choice='robust'
    )
params_table_fe['method'] = 'FE'

mod_timefe, res_timefe, params_table_timefe, joint_teststats_timefe, reg_det_timefe = \
    fe_reg(
        df=df,
        y_col=outcome_choice,
        x_cols=[income_choice],
        i_col='cohort_code',
        t_col='year',
        fixed_effects=False,
        time_effects=True,
        cov_choice='robust'
    )
params_table_timefe['method'] = 'TimeFE'

mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = \
    re_reg(
        df=df,
        y_col=outcome_choice,
        x_cols=[income_choice],
        i_col='cohort_code',
        t_col='year',
        cov_choice='robust'
    )
params_table_re['method'] = 'RE'

params_table_cohort = pd.concat([pd.DataFrame(params_table_fe.loc[income_choice]).transpose(),
                                 pd.DataFrame(params_table_timefe.loc[income_choice]).transpose(),
                                 pd.DataFrame(params_table_re.loc[income_choice]).transpose()], axis=0)
dfi.export(params_table_cohort,
           'output/params_table_overall_mean' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + '.png')
telsendimg(
    conf=tel_config,
    path='output/params_table_overall_mean' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + '.png',
    cap='params_table_overall_mean' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix
)

# Execute individual-level (no FD option)

mod_ind_ols, res_ind_ols, params_table_ind_ols, joint_teststats_ind_ols, reg_det_ind_ols = \
    reg_ols(
        df=df_ind,
        eqn=outcome_choice + ' ~ ' + income_choice + ' + ' +
            'C(state) + urban + C(education) + C(ethnicity) + ' +
            'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
            'C(child_group) + male + C(birth_year_group) + C(marriage) + ' +
            'C(emp_status) + C(industry) + C(occupation) + C(year)'
    )
dfi.export(pd.DataFrame(params_table_ind_ols.loc[income_choice]),
           'output/params_table_ind_ols_overall' + '_' + outcome_choice + '_' + income_choice + '.png',
           fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)  # to overcome mar2023 error
telsendimg(
    conf=tel_config,
    path='output/params_table_ind_ols_overall' + '_' + outcome_choice + '_' + income_choice + '.png',
    cap='params_table_ind_ols_overall' + '_' + outcome_choice + '_' + income_choice
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_reg_overall: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
