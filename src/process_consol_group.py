import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles
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
path_2009 = './data/hies_2009/'
path_2014 = './data/hies_2014/'
path_2016 = './data/hies_2016/'
path_2019 = './data/hies_2019/'

# I --- Load processed vintages
df_09 = pd.read_parquet(path_2009 + 'hies_2009_consol.parquet')
df_14 = pd.read_parquet(path_2014 + 'hies_2014_consol.parquet')
df_16 = pd.read_parquet(path_2016 + 'hies_2016_consol.parquet')
df_19 = pd.read_parquet(path_2019 + 'hies_2019_consol.parquet')

# II --- Identify common columns
common_cols_14_19 = list(set(df_14.columns) & set(df_16.columns) & set(df_19.columns))
common_col_09_19 = list(set(df_09.columns) & set(df_14.columns) & set(df_16.columns) & set(df_19.columns))

# III --- Pre-merger cleaning

# Harmonise ethnicity levels (to lowest; 2014)
df_16.loc[~(df_19['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'
df_19.loc[~(df_19['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'

# Not useful
for i in ['id', 'svy_weight']:
    del df_09[i]
    del df_14[i]
    del df_16[i]
    del df_19[i]

# IV.A --- Merger (group-level; 2014 - 2019)

# Aggregation rule
# dict_agg_rule = \
#     {
#         'salaried_wages': np.mean,
#         'other_wages': np.mean,
#         'asset_income': np.mean,
#         'gross_transfers': np.mean,
#         'gross_income': np.mean,
#         'gross_margin': np.mean,
#         'income_gen_members': np.mean,
#         'cons_01_12': np.mean,
#         'cons_01_13': np.mean,
#         'cons_01': np.mean,
#         'cons_02': np.mean,
#         'cons_03': np.mean,
#         'cons_04': np.mean,
#         'cons_05': np.mean,
#         'cons_06': np.mean,
#         'cons_07': np.mean,
#         'cons_08': np.mean,
#         'cons_09': np.mean,
#         'cons_10': np.mean,
#         'cons_11': np.mean,
#         'cons_12': np.mean,
#         'cons_13': np.mean,
#     }

# Observables to be grouped on
col_groups = \
    [
        'state',
        'urban',
        'education',
        'ethnicity', 'malaysian',
        'income_gen_members',
        'adolescent', 'child',
        'male',
        # 'age',
        'age_group',
        'marriage',
        'emp_status',
        'industry',
        'occupation',
        # 'hh_size',
    ]

# Delete redundant columns
for i in ['age', 'hh_size',]:
    del df_14[i]
    del df_16[i]
    del df_19[i]
for i in ['monthly_income', 'net_income', 'net_transfers', 'net_margin']:
    del df_16[i]
    del df_19[i]

# Groupby operation
df_14_agg = df_14.groupby(col_groups).mean(numeric_only=True).reset_index()
df_16_agg = df_16.groupby(col_groups).mean(numeric_only=True).reset_index()
df_19_agg = df_19.groupby(col_groups).mean(numeric_only=True).reset_index()

# Generate time identifiers
df_14_agg['year'] = 2014
df_16_agg['year'] = 2016
df_19_agg['year'] = 2019

# Merge
df_agg = pd.concat([df_14_agg, df_16_agg, df_19_agg], axis=0)

# Sort
df_agg = df_agg.sort_values(by=col_groups + ['year']).reset_index(drop=True)

# V --- Output
df_agg.to_parquet(path_data + 'hies_consol_agg.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_consol: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
