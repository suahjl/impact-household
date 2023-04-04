# Merging at the group-level

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

df_14_hhbasis = pd.read_parquet(path_2014 + 'hies_2014_consol_hhbasis_trimmedoutliers.parquet')
df_16_hhbasis = pd.read_parquet(path_2016 + 'hies_2016_consol_hhbasis_trimmedoutliers.parquet')
df_19_hhbasis = pd.read_parquet(path_2019 + 'hies_2019_consol_hhbasis_trimmedoutliers.parquet')

df_09_full = pd.read_parquet(path_2009 + 'hies_2009_consol.parquet')
df_14_full = pd.read_parquet(path_2014 + 'hies_2014_consol.parquet')
df_16_full = pd.read_parquet(path_2016 + 'hies_2016_consol.parquet')
df_19_full = pd.read_parquet(path_2019 + 'hies_2019_consol.parquet')

df_14_full_hhbasis = pd.read_parquet(path_2014 + 'hies_2014_consol_hhbasis.parquet')
df_16_full_hhbasis = pd.read_parquet(path_2016 + 'hies_2016_consol_hhbasis.parquet')
df_19_full_hhbasis = pd.read_parquet(path_2019 + 'hies_2019_consol_hhbasis.parquet')

# II --- Identify common columns
common_cols_14_19 = list(set(df_14.columns) & set(df_16.columns) & set(df_19.columns))
common_col_09_19 = list(set(df_09.columns) & set(df_14.columns) & set(df_16.columns) & set(df_19.columns))

# III --- Pre-merger cleaning

# Harmonise ethnicity levels (to lowest; 2014)
df_16.loc[~(df_16['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'
df_19.loc[~(df_19['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'

df_16_full.loc[~(df_16_full['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'
df_19_full.loc[~(df_19_full['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'

df_16_full_hhbasis.loc[~(df_16_full_hhbasis['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'
df_19_full_hhbasis.loc[~(df_19_full_hhbasis['ethnicity'] == 'bumiputera'), 'ethnicity'] = 'non_bumiputera'

# Not useful
for i in ['id', 'svy_weight']:
    del df_09[i]
    del df_14[i]
    del df_16[i]
    del df_19[i]

    del df_14_hhbasis[i]
    del df_16_hhbasis[i]
    del df_19_hhbasis[i]

    del df_09_full[i]
    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]

    del df_14_full_hhbasis[i]
    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]


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

gen_igm_group(data=df_14_hhbasis)
gen_igm_group(data=df_16_hhbasis)
gen_igm_group(data=df_19_hhbasis)

gen_igm_group(data=df_14_full)
gen_igm_group(data=df_16_full)
gen_igm_group(data=df_19_full)

gen_igm_group(data=df_14_full_hhbasis)
gen_igm_group(data=df_16_full_hhbasis)
gen_igm_group(data=df_19_full_hhbasis)


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

gen_child_group(data=df_14_hhbasis)
gen_child_group(data=df_16_hhbasis)
gen_child_group(data=df_19_hhbasis)

gen_child_group(data=df_14_full)
gen_child_group(data=df_16_full)
gen_child_group(data=df_19_full)

gen_child_group(data=df_14_full_hhbasis)
gen_child_group(data=df_16_full_hhbasis)
gen_child_group(data=df_19_full_hhbasis)


# adolescents (1, 2, 3+)
def gen_adolescent_group(data):
    data.loc[data['adolescent'] == 0, 'adolescent_group'] = '0'
    data.loc[data['adolescent'] == 1, 'adolescent_group'] = '1'
    data.loc[data['adolescent'] >= 2, 'adolescent_group'] = '2+'
    del data['adolescent']


gen_adolescent_group(data=df_14)
gen_adolescent_group(data=df_16)
gen_adolescent_group(data=df_19)

gen_adolescent_group(data=df_14_hhbasis)
gen_adolescent_group(data=df_16_hhbasis)
gen_adolescent_group(data=df_19_hhbasis)

gen_adolescent_group(data=df_14_full)
gen_adolescent_group(data=df_16_full)
gen_adolescent_group(data=df_19_full)

gen_adolescent_group(data=df_14_full_hhbasis)
gen_adolescent_group(data=df_16_full_hhbasis)
gen_adolescent_group(data=df_19_full_hhbasis)


# collapse marriage groups
def collapse_marriage(data):
    data.loc[((data['marriage'] == 'separated') |
              (data['marriage'] == 'divorced') |
              (data['marriage'] == 'never') |
              (data['marriage'] == 'widowed')), 'marriage'] = 'single'


collapse_marriage(data=df_14)
collapse_marriage(data=df_16)
collapse_marriage(data=df_19)

collapse_marriage(data=df_14_hhbasis)
collapse_marriage(data=df_16_hhbasis)
collapse_marriage(data=df_19_hhbasis)

collapse_marriage(data=df_14_full)
collapse_marriage(data=df_16_full)
collapse_marriage(data=df_19_full)

collapse_marriage(data=df_14_full_hhbasis)
collapse_marriage(data=df_16_full_hhbasis)
collapse_marriage(data=df_19_full_hhbasis)


# collapse education
def collapse_education(data):
    data.loc[((data['education'] == 'stpm') |
              (data['education'] == 'spm') |
              (data['education'] == 'pmr')), 'education'] = 'school'


collapse_education(data=df_14)
collapse_education(data=df_16)
collapse_education(data=df_19)

collapse_education(data=df_14_hhbasis)
collapse_education(data=df_16_hhbasis)
collapse_education(data=df_19_hhbasis)

collapse_education(data=df_14_full)
collapse_education(data=df_16_full)
collapse_education(data=df_19_full)

collapse_education(data=df_14_full_hhbasis)
collapse_education(data=df_16_full_hhbasis)
collapse_education(data=df_19_full_hhbasis)


# collapse emp_status
def collapse_emp_status(data):
    data.loc[((data['emp_status'] == 'housespouse') |
              (data['emp_status'] == 'unemployed') |
              (data['emp_status'] == 'unpaid_fam') |
              (data['emp_status'] == 'child_not_at_work')), 'emp_status'] = 'no_paid_work'


collapse_emp_status(data=df_14)
collapse_emp_status(data=df_16)
collapse_emp_status(data=df_19)

collapse_emp_status(data=df_14_hhbasis)
collapse_emp_status(data=df_16_hhbasis)
collapse_emp_status(data=df_19_hhbasis)

collapse_emp_status(data=df_14_full)
collapse_emp_status(data=df_16_full)
collapse_emp_status(data=df_19_full)

collapse_emp_status(data=df_14_full_hhbasis)
collapse_emp_status(data=df_16_full_hhbasis)
collapse_emp_status(data=df_19_full_hhbasis)


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

gen_age_group(data=df_14_hhbasis, aggregation=2)
gen_age_group(data=df_16_hhbasis, aggregation=2)
gen_age_group(data=df_19_hhbasis, aggregation=2)

gen_age_group(data=df_14_full, aggregation=2)
gen_age_group(data=df_16_full, aggregation=2)
gen_age_group(data=df_19_full, aggregation=2)

gen_age_group(data=df_14_full_hhbasis, aggregation=2)
gen_age_group(data=df_16_full_hhbasis, aggregation=2)
gen_age_group(data=df_19_full_hhbasis, aggregation=2)


# Birth year groups
def gen_birth_year_group(data, aggregation):
    if aggregation == 1:
        data.loc[(data['birth_year'] >= 1990), 'birth_year_group'] = '1990+'
        data.loc[((data['birth_year'] >= 1980) & (data['birth_year'] <= 1989)), 'birth_year_group'] = '1980s'
        data.loc[((data['birth_year'] >= 1970) & (data['birth_year'] <= 1979)), 'birth_year_group'] = '1970s'
        data.loc[((data['birth_year'] >= 1960) & (data['birth_year'] <= 1969)), 'birth_year_group'] = '1960s'
        data.loc[((data['birth_year'] >= 1950) & (data['birth_year'] <= 1959)), 'birth_year_group'] = '1950s'
        data.loc[(data['birth_year'] <= 1949), 'birth_year_group'] = '1949-'
    elif aggregation == 2:
        data.loc[(data['birth_year'] >= 1980), 'birth_year_group'] = '1980+'
        data.loc[((data['birth_year'] >= 1960) & (data['birth_year'] <= 1979)), 'birth_year_group'] = '1960_79'
        data.loc[(data['birth_year'] <= 1959), 'birth_year_group'] = '1959-'
    del data['birth_year']


gen_birth_year_group(data=df_14, aggregation=2)
gen_birth_year_group(data=df_16, aggregation=2)
gen_birth_year_group(data=df_19, aggregation=2)

gen_birth_year_group(data=df_14_hhbasis, aggregation=2)
gen_birth_year_group(data=df_16_hhbasis, aggregation=2)
gen_birth_year_group(data=df_19_hhbasis, aggregation=2)

gen_birth_year_group(data=df_14_full, aggregation=2)
gen_birth_year_group(data=df_16_full, aggregation=2)
gen_birth_year_group(data=df_19_full, aggregation=2)

gen_birth_year_group(data=df_14_full_hhbasis, aggregation=2)
gen_birth_year_group(data=df_16_full_hhbasis, aggregation=2)
gen_birth_year_group(data=df_19_full_hhbasis, aggregation=2)

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
        'birth_year_group',
        'marriage',
        'emp_status',
        'industry',
        'occupation'
    ]

# Delete redundant columns; keep hh_size for hhbasis dataset
for i in ['hh_size']:
    del df_14[i]
    del df_16[i]
    del df_19[i]

    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]

for i in ['monthly_income', 'net_income', 'net_transfers', 'net_margin']:
    del df_16[i]
    del df_19[i]

    del df_16_hhbasis[i]
    del df_19_hhbasis[i]

    del df_16_full[i]
    del df_19_full[i]

    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]


# Cohort merger loop + output
def gen_pseudopanel(data1, data2, data3, list_cols_cohort, use_mean, use_quantile, quantile_choice, file_suffix):
    # Groupby operation
    if use_mean and not use_quantile:
        df1_agg = data1.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
        df2_agg = data2.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
        df3_agg = data3.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    elif use_quantile and not use_mean:
        df1_agg = data1.groupby(list_cols_cohort).quantile(numeric_only=True, q=quantile_choice).reset_index()
        df2_agg = data2.groupby(list_cols_cohort).quantile(numeric_only=True, q=quantile_choice).reset_index()
        df3_agg = data3.groupby(list_cols_cohort).quantile(numeric_only=True, q=quantile_choice).reset_index()
    elif use_mean and use_quantile:
        raise NotImplementedError('Use either mean or quantiles')
    elif not use_mean and not use_quantile:
        raise NotImplementedError('Use either mean or quantiles')
    # Generate time identifiers
    df1_agg['_time'] = 1
    df2_agg['_time'] = 2
    df3_agg['_time'] = 3
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(df2_agg[list_cols_cohort], on=list_cols_cohort, how='inner')
    groups_balanced = groups_balanced[list_cols_cohort].merge(df3_agg[list_cols_cohort], on=list_cols_cohort,
                                                              how='inner')
    groups_balanced['balanced'] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how='left')
    df_agg_balanced = df_agg_balanced[df_agg_balanced['balanced'] == 1]
    del df_agg_balanced['balanced']
    df_agg_balanced = df_agg_balanced.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Save to local
    df_agg.to_parquet(path_data + 'hies_consol_agg_' + file_suffix + '.parquet')
    df_agg_balanced.to_parquet(path_data + 'hies_consol_agg_balanced_' + file_suffix + '.parquet')
    # Output
    return df_agg, df_agg_balanced


def gen_pseudopanel_quantile_fixed_axis(
        data1,
        data2,
        data3,
        list_cols_cohort,
        fixed_axis,
        quantile_choice,
        file_suffix
):
    # Prelims
    df1 = data1.copy()
    df2 = data2.copy()
    df3 = data3.copy()
    # Create reference point on fixed axis
    df1_ref = data1.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=quantile_choice, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_fixed_axis'})
    df2_ref = data2.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=quantile_choice, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_fixed_axis'})
    df3_ref = data3.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=quantile_choice, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_fixed_axis'})
    # Merge back
    df1 = df1.merge(df1_ref, on=list_cols_cohort, how='left')
    df2 = df2.merge(df2_ref, on=list_cols_cohort, how='left')
    df3 = df3.merge(df3_ref, on=list_cols_cohort, how='left')
    # Keep rows where quantile ref point match
    df1 = df1[df1[fixed_axis] == df1['_fixed_axis']]
    df2 = df2[df2[fixed_axis] == df2['_fixed_axis']]
    df3 = df3[df3[fixed_axis] == df3['_fixed_axis']]
    # Collapse
    df1_agg = df1.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    df2_agg = df2.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    df3_agg = df3.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    # Generate time identifiers
    df1_agg['_time'] = 1
    df2_agg['_time'] = 2
    df3_agg['_time'] = 3
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(df2_agg[list_cols_cohort], on=list_cols_cohort, how='inner')
    groups_balanced = groups_balanced[list_cols_cohort].merge(df3_agg[list_cols_cohort], on=list_cols_cohort,
                                                              how='inner')
    groups_balanced['balanced'] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how='left')
    df_agg_balanced = df_agg_balanced[df_agg_balanced['balanced'] == 1]
    del df_agg_balanced['balanced']
    df_agg_balanced = df_agg_balanced.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Delete fixed axes
    del df_agg['_fixed_axis']
    del df_agg_balanced['_fixed_axis']
    # Save to local
    df_agg.to_parquet(path_data + 'hies_consol_agg_' + file_suffix + '.parquet')
    df_agg_balanced.to_parquet(path_data + 'hies_consol_agg_balanced_' + file_suffix + '.parquet')
    # Output
    return df_agg, df_agg_balanced


def gen_pseudopanel_distgroup_fixed_axis(
        data1,
        data2,
        data3,
        list_cols_cohort,
        fixed_axis,
        distbounds,
        file_suffix
):
    # Prelims
    df1 = data1.copy()
    df2 = data2.copy()
    df3 = data3.copy()
    q_lb = distbounds[0]  # lower bound of distribution
    q_ub = distbounds[1]  # upper bound of distribution
    # Create reference points (LB and UB) on fixed axis
    df1_lb = data1.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_lb, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_lb_fixed_axis'})
    df1_ub = data1.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_ub, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_ub_fixed_axis'})
    df2_lb = data2.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_lb, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_lb_fixed_axis'})
    df2_ub = data2.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_ub, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_ub_fixed_axis'})
    df3_lb = data3.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_lb, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_lb_fixed_axis'})
    df3_ub = data3.groupby(list_cols_cohort)[fixed_axis] \
        .quantile(numeric_only=True, q=q_ub, interpolation='nearest') \
        .reset_index().rename(columns={fixed_axis: '_ub_fixed_axis'})
    # Merge back
    df1 = df1.merge(df1_lb, on=list_cols_cohort, how='left')
    df1 = df1.merge(df1_ub, on=list_cols_cohort, how='left')
    df2 = df2.merge(df2_lb, on=list_cols_cohort, how='left')
    df2 = df2.merge(df2_ub, on=list_cols_cohort, how='left')
    df3 = df3.merge(df3_lb, on=list_cols_cohort, how='left')
    df3 = df3.merge(df3_ub, on=list_cols_cohort, how='left')
    # Keep rows where fixed axis observations fall within range
    df1 = df1[(df1[fixed_axis] >= df1['_lb_fixed_axis']) & (df1[fixed_axis] <= df1['_ub_fixed_axis'])]
    df2 = df2[(df2[fixed_axis] >= df2['_lb_fixed_axis']) & (df2[fixed_axis] <= df2['_ub_fixed_axis'])]
    df3 = df3[(df3[fixed_axis] >= df3['_lb_fixed_axis']) & (df3[fixed_axis] <= df3['_ub_fixed_axis'])]
    # Collapse
    df1_agg = df1.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    df2_agg = df2.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    df3_agg = df3.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
    # Generate time identifiers
    df1_agg['_time'] = 1
    df2_agg['_time'] = 2
    df3_agg['_time'] = 3
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(df2_agg[list_cols_cohort], on=list_cols_cohort, how='inner')
    groups_balanced = groups_balanced[list_cols_cohort].merge(df3_agg[list_cols_cohort], on=list_cols_cohort,
                                                              how='inner')
    groups_balanced['balanced'] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how='left')
    df_agg_balanced = df_agg_balanced[df_agg_balanced['balanced'] == 1]
    del df_agg_balanced['balanced']
    df_agg_balanced = df_agg_balanced.sort_values(by=list_cols_cohort + ['_time']).reset_index(drop=True)
    # Delete fixed axes
    del df_agg['_lb_fixed_axis']
    del df_agg['_ub_fixed_axis']
    del df_agg_balanced['_lb_fixed_axis']
    del df_agg_balanced['_ub_fixed_axis']
    # Save to local
    df_agg.to_parquet(path_data + 'hies_consol_agg_' + file_suffix + '.parquet')
    df_agg_balanced.to_parquet(path_data + 'hies_consol_agg_balanced_' + file_suffix + '.parquet')
    # Output
    return df_agg, df_agg_balanced


df_agg_mean, df_agg_balanced_mean = gen_pseudopanel(
    data1=df_14,
    data2=df_16,
    data3=df_19,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean'
)

# list_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# list_suffixes = ['10p', '20p', '30p', '40p', '50p', '60p', '70p', '80p', '90p', '100p']
# for quantile, suffix in tqdm(zip(list_quantiles, list_suffixes)):
    # Quantiles across all income and consumption categories
    # df_agg_quantile, df_agg_balanced_quantile = gen_pseudopanel(
    #     data1=df_14,
    #     data2=df_16,
    #     data3=df_19,
    #     list_cols_cohort=col_groups,
    #     use_mean=False,
    #     use_quantile=True,
    #     quantile_choice=quantile,
    #     file_suffix=suffix
    # )

    # Use fixed axis on gross income when generating quantile-specific cohort panel
    # df_agg_quantile, df_agg_balanced_quantile = gen_pseudopanel_quantile_fixed_axis(
    #     data1=df_14,
    #     data2=df_16,
    #     data3=df_19,
    #     fixed_axis='gross_income',
    #     list_cols_cohort=col_groups,
    #     quantile_choice=quantile,
    #     file_suffix=suffix
    # )

# list_distbounds = [
#     [0, 0.1],
#     [0.1, 0.2],
#     [0.2, 0.3],
#     [0.3, 0.4],
#     [0.4, 0.5],
#     [0.5, 0.6],
#     [0.6, 0.7],
#     [0.7, 0.8],
#     [0.8, 0.9],
#     [0.9, 1]
# ]
list_distbounds = [
    [0, 0.2],
    [0.2, 0.4],
    [0.4, 0.6],
    [0.6, 0.8],
    [0.8, 1]
]
list_suffixes = ['20p', '40p', '60p', '80p', '100p']
# list_suffixes = ['10p', '20p', '30p', '40p', '50p', '60p', '70p', '80p', '90p', '100p']
for distbound, suffix in tqdm(zip(list_distbounds, list_suffixes)):
    # Use fixed axis on gross income when generating 10pp width buckets cohort panel
    df_agg_quantile, df_agg_balanced_quantile = gen_pseudopanel_distgroup_fixed_axis(
        data1=df_14,
        data2=df_16,
        data3=df_19,
        fixed_axis='gross_income',
        list_cols_cohort=col_groups,
        distbounds=distbound,
        file_suffix=suffix
    )

# Individual pooled data + output
df_14['year'] = 2014
df_16['year'] = 2016
df_19['year'] = 2019
df_ind = pd.concat([df_14, df_16, df_19], axis=0)
df_ind = df_ind.sort_values(by=col_groups + ['year']).reset_index(drop=True)
df_ind.to_parquet(path_data + 'hies_consol_ind.parquet')

# Individual pooled data + output (household basis)
df_14_hhbasis['year'] = 2014
df_16_hhbasis['year'] = 2016
df_19_hhbasis['year'] = 2019
df_ind_hhbasis = pd.concat([df_14_hhbasis, df_16_hhbasis, df_19_hhbasis], axis=0)
df_ind_hhbasis = df_ind_hhbasis.sort_values(by=col_groups + ['year']).reset_index(drop=True)
df_ind_hhbasis.to_parquet(path_data + 'hies_consol_ind_hhbasis.parquet')

# Individual pooled data + output (with outliers)
df_14_full['year'] = 2014
df_16_full['year'] = 2016
df_19_full['year'] = 2019
df_ind_full = pd.concat([df_14_full, df_16_full, df_19_full], axis=0)
df_ind_full = df_ind_full.sort_values(by=col_groups + ['year']).reset_index(drop=True)
df_ind_full.to_parquet(path_data + 'hies_consol_ind_full.parquet')

# Individual pooled data + output (with outliers and household basis)
df_14_full_hhbasis['year'] = 2014
df_16_full_hhbasis['year'] = 2016
df_19_full_hhbasis['year'] = 2019
df_ind_full_hhbasis = pd.concat([df_14_full_hhbasis, df_16_full_hhbasis, df_19_full_hhbasis], axis=0)
df_ind_full_hhbasis = df_ind_full_hhbasis.sort_values(by=col_groups + ['year']).reset_index(drop=True)
df_ind_full_hhbasis.to_parquet(path_data + 'hies_consol_ind_full_hhbasis.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_consol_group: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
