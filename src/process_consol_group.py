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
df_09 = pd.read_parquet(path_2009 + 'hies_2009_consol_trimmedoutliers.parquet')
df_14 = pd.read_parquet(path_2014 + 'hies_2014_consol_trimmedoutliers.parquet')
df_16 = pd.read_parquet(path_2016 + 'hies_2016_consol_trimmedoutliers.parquet')
df_19 = pd.read_parquet(path_2019 + 'hies_2019_consol_trimmedoutliers.parquet')

# II --- Identify common columns
common_cols_14_19 = list(set(df_14.columns) & set(df_16.columns) & set(df_19.columns))
common_col_09_19 = list(set(df_09.columns) & set(df_14.columns) & set(df_16.columns) & set(df_19.columns))

# III --- Pre-merger cleaning

# Harmonise ethnicity levels (to lowest; 2014)
df_16.loc[~(df_16['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'
df_19.loc[~(df_19['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'

# Not useful
for i in ['id', 'svy_weight']:
    del df_09[i]
    del df_14[i]
    del df_16[i]
    del df_19[i]


# Buckets for number of income_gen_members, child, and adolescents, and marriage status (age done separately)
# income_gen_members (1, 2, 3+)
def gen_igm_group(data):
    data.loc[data['income_gen_members'] == 1, 'income_gen_members_group'] = '1'
    data.loc[data['income_gen_members'] == 2, 'income_gen_members_group'] = '2'
    data.loc[data['income_gen_members'] >= 3, 'income_gen_members_group'] = '3+'
    del data['income_gen_members']


gen_igm_group(data=df_14)
gen_igm_group(data=df_16)
gen_igm_group(data=df_19)


# child (1, 2, 3+)
def gen_child_group(data):
    data.loc[data['child'] == 0, 'child_group'] = '0'
    data.loc[data['child'] == 1, 'child_group'] = '1'
    data.loc[data['child'] == 2, 'child_group'] = '2'
    data.loc[data['child'] >= 3, 'child_group'] = '3+'
    del data['child']


gen_child_group(data=df_14)
gen_child_group(data=df_16)
gen_child_group(data=df_19)


# adolescents (1, 2, 3+)
def gen_adolescent_group(data):
    data.loc[data['adolescent'] == 0, 'adolescent_group'] = '0'
    data.loc[data['adolescent'] == 1, 'adolescent_group'] = '1'
    data.loc[data['adolescent'] >= 2, 'adolescent_group'] = '2+'
    del data['adolescent']


gen_adolescent_group(data=df_14)
gen_adolescent_group(data=df_16)
gen_adolescent_group(data=df_19)


# collapse marriage groups
def collapse_marriage(data):
    data.loc[((data['marriage'] == 'separated') |
              (data['marriage'] == 'divorced') |
              (data['marriage'] == 'never') |
              (data['marriage'] == 'widowed')), 'marriage'] = 'single'


collapse_marriage(data=df_14)
collapse_marriage(data=df_16)
collapse_marriage(data=df_19)


# collapse education
def collapse_education(data):
    data.loc[((data['education'] == 'stpm') |
              (data['education'] == 'spm') |
              (data['education'] == 'pmr')), 'education'] = 'school'


collapse_education(data=df_14)
collapse_education(data=df_16)
collapse_education(data=df_19)


# collapse emp_status
def collapse_emp_status(data):
    data.loc[((data['emp_status'] == 'housespouse') |
              (data['emp_status'] == 'unemployed') |
              (data['emp_status'] == 'unpaid_fam') |
              (data['emp_status'] == 'child_not_at_work')), 'emp_status'] = 'no_paid_work'


collapse_emp_status(data=df_14)
collapse_emp_status(data=df_16)
collapse_emp_status(data=df_19)

# collapse industry
# def collapse_industry(data):
#     data.loc[((data['']) |
#               ())]


# age groups
def gen_age_group(data, aggregation):
    if aggregation == 1:
        data.loc[(data['age'] <= 29), 'age_group'] = '0_29'
        data.loc[((data['age'] >= 30) & (data['age'] <= 39)), 'age_group'] = '30_39'
        data.loc[((data['age'] >= 40) & (data['age'] <= 49)), 'age_group'] = '40_49'
        data.loc[((data['age'] >= 50) & (data['age'] <= 59)), 'age_group'] = '50_59'
        data.loc[((data['age'] >= 60) & (data['age'] <= 69)), 'age_group'] = '60_69'
        data.loc[(data['age'] >= 70), 'age_group'] = '70+'
    elif aggregation == 2:
        data.loc[(data['age'] <= 39), 'age_group'] = '0_39'
        data.loc[((data['age'] >= 40) & (data['age'] <= 59)), 'age_group'] = '40_59'
        data.loc[(data['age'] >= 60), 'age_group'] = '60+'

    del data['age']


gen_age_group(data=df_14, aggregation=2)
gen_age_group(data=df_16, aggregation=2)
gen_age_group(data=df_19, aggregation=2)

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

# Delete redundant columns
for i in ['hh_size']:
    del df_14[i]
    del df_16[i]
    del df_19[i]
for i in ['monthly_income', 'net_income', 'net_transfers', 'net_margin']:
    del df_16[i]
    del df_19[i]

# groupby operation
df_14_agg = df_14.groupby(col_groups).mean(numeric_only=True).reset_index()
df_16_agg = df_16.groupby(col_groups).mean(numeric_only=True).reset_index()
df_19_agg = df_19.groupby(col_groups).mean(numeric_only=True).reset_index()

# Generate time identifiers
df_14_agg['year'] = 2014
df_16_agg['year'] = 2016
df_19_agg['year'] = 2019

df_14['year'] = 2014
df_16['year'] = 2016
df_19['year'] = 2019

# Merge (unbalanced)
df_agg = pd.concat([df_14_agg, df_16_agg, df_19_agg], axis=0)
df_agg = df_agg.sort_values(by=col_groups + ['year']).reset_index(drop=True)

# Merge (balanced)
groups_balanced = df_14_agg[col_groups].merge(df_16_agg[col_groups], on=col_groups, how='inner')
groups_balanced = groups_balanced[col_groups].merge(df_19_agg[col_groups], on=col_groups, how='inner')
groups_balanced['balanced'] = 1
df_agg_balanced = df_agg.merge(groups_balanced, on=col_groups, how='left')
df_agg_balanced = df_agg_balanced[df_agg_balanced['balanced'] == 1]
del df_agg_balanced['balanced']
df_agg_balanced = df_agg_balanced.sort_values(by=col_groups + ['year']).reset_index(drop=True)

# Merge (individual-pooled)
df_ind = pd.concat([df_14, df_16, df_19], axis=0)
df_ind = df_ind.sort_values(by=col_groups + ['year']).reset_index(drop=True)

# V --- Output
df_agg.to_parquet(path_data + 'hies_consol_agg.parquet')
df_agg_balanced.to_parquet(path_data + 'hies_consol_agg_balanced.parquet')
df_ind.to_parquet(path_data + 'hies_consol_ind.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_consol: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
