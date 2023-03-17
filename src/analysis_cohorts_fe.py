import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols
from tabulate import tabulate
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
        'age_group',
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
    df[i] = np.log(df[i])
    df_ind[i] = np.log(df_ind[i])

# III --- Estimation
mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
    fe_reg(
        df=df,
        y_col='cons_01_12',
        x_cols=['gross_income'],
        i_col='cohort_code',
        t_col='year',
        fixed_effects=True,
        time_effects=False,
        cov_choice='robust'
    )

mod_ols, res_ols, params_table_ols, joint_teststats_ols, reg_det_ols = \
    reg_ols(
        df=df_ind,
        eqn='cons_01_12 ~ gross_income + ' +
            'C(state) + urban + C(education) + C(ethnicity) + ' +
            'malaysian + C(income_gen_members_group) + C(adolescent_group) +' +
            'C(child_group) + male + C(age_group) + C(marriage) + ' +
            'C(emp_status) + C(industry) + C(occupation)'
    )

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_cohorts_fe: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
