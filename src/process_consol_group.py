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
hhbasis_cohorts_with_hhsize = ast.literal_eval(os.getenv('HHBASIS_COHORTS_WITH_HHSIZE'))

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
    # del data['income_gen_members']


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


# non-working adult females (0, 1, 2, 3+)
def gen_nwaf_group(data):
    data.loc[data['non_working_adult_females'] == 0, 'non_working_adult_females_group'] = '0'
    data.loc[data['non_working_adult_females'] == 1, 'non_working_adult_females_group'] = '1'
    data.loc[data['non_working_adult_females'] == 2, 'non_working_adult_females_group'] = '2'
    data.loc[data['non_working_adult_females'] >= 3, 'non_working_adult_females_group'] = '3+'
    # del data['non_working_adult_females']


gen_nwaf_group(data=df_14)
gen_nwaf_group(data=df_16)
gen_nwaf_group(data=df_19)

gen_nwaf_group(data=df_14_hhbasis)
gen_nwaf_group(data=df_16_hhbasis)
gen_nwaf_group(data=df_19_hhbasis)

gen_nwaf_group(data=df_14_full)
gen_nwaf_group(data=df_16_full)
gen_nwaf_group(data=df_19_full)

gen_nwaf_group(data=df_14_full_hhbasis)
gen_nwaf_group(data=df_16_full_hhbasis)
gen_nwaf_group(data=df_19_full_hhbasis)


# working adult females (0, 1, 2, 3+)
def gen_waf_group(data):
    data.loc[data['working_adult_females'] == 0, 'working_adult_females_group'] = '0'
    data.loc[data['working_adult_females'] == 1, 'working_adult_females_group'] = '1'
    data.loc[data['working_adult_females'] == 2, 'working_adult_females_group'] = '2'
    data.loc[data['working_adult_females'] >= 3, 'working_adult_females_group'] = '3+'
    # del data['working_adult_females']


gen_waf_group(data=df_14)
gen_waf_group(data=df_16)
gen_waf_group(data=df_19)

gen_waf_group(data=df_14_hhbasis)
gen_waf_group(data=df_16_hhbasis)
gen_waf_group(data=df_19_hhbasis)

gen_waf_group(data=df_14_full)
gen_waf_group(data=df_16_full)
gen_waf_group(data=df_19_full)

gen_waf_group(data=df_14_full_hhbasis)
gen_waf_group(data=df_16_full_hhbasis)
gen_waf_group(data=df_19_full_hhbasis)


# child (1, 2, 3+)
def gen_child_group(data):
    data.loc[data['child'] == 0, 'child_group'] = '0'
    data.loc[data['child'] == 1, 'child_group'] = '1'
    data.loc[data['child'] == 2, 'child_group'] = '2'
    data.loc[data['child'] >= 3, 'child_group'] = '3+'
    # del data['child']


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
    data.loc[data['adolescent'] == 2, 'adolescent_group'] = '2'
    data.loc[data['adolescent'] >= 3, 'adolescent_group'] = '3+'
    # del data['adolescent']


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


# elderly group (1, 2, 3+)
def gen_elderly_group(data):
    data.loc[data['elderly'] == 0, 'elderly_group'] = '0'
    data.loc[data['elderly'] == 1, 'elderly_group'] = '1'
    data.loc[data['elderly'] == 2, 'elderly_group'] = '2'
    data.loc[data['elderly'] >= 3, 'elderly_group'] = '3+'
    # del data['elderly']


gen_elderly_group(data=df_14)
gen_elderly_group(data=df_16)
gen_elderly_group(data=df_19)

gen_elderly_group(data=df_14_hhbasis)
gen_elderly_group(data=df_16_hhbasis)
gen_elderly_group(data=df_19_hhbasis)

gen_elderly_group(data=df_14_full)
gen_elderly_group(data=df_16_full)
gen_elderly_group(data=df_19_full)

gen_elderly_group(data=df_14_full_hhbasis)
gen_elderly_group(data=df_16_full_hhbasis)
gen_elderly_group(data=df_19_full_hhbasis)


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
    # del data['age']


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
    # del data['birth_year']


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


# Household size groups (only for household basis)
def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 8, 'hh_size_group'] = '8+'
    data['hh_size_group'] = data['hh_size_group'].astype('str')


gen_hh_size_group(data=df_14_hhbasis)
gen_hh_size_group(data=df_16_hhbasis)
gen_hh_size_group(data=df_19_hhbasis)

gen_hh_size_group(data=df_14_full_hhbasis)
gen_hh_size_group(data=df_16_full_hhbasis)
gen_hh_size_group(data=df_19_full_hhbasis)

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

# Create copy of dataframes with base columns that have been transformed into categoricals
df_14_withbase = df_14.copy()
df_16_withbase = df_16.copy()
df_19_withbase = df_19.copy()
df_14_hhbasis_withbase = df_14_hhbasis.copy()
df_16_hhbasis_withbase = df_16_hhbasis.copy()
df_19_hhbasis_withbase = df_19_hhbasis.copy()
df_14_full_withbase = df_14_full.copy()
df_16_full_withbase = df_16_full.copy()
df_19_full_withbase = df_19_full.copy()
df_14_full_hhbasis_withbase = df_14_full_hhbasis.copy()
df_16_full_hhbasis_withbase = df_16_full_hhbasis.copy()
df_19_full_hhbasis_withbase = df_19_full_hhbasis.copy()

# Save copy of dataframes with base columns that have been transformed into categoricals
df_14_withbase.to_parquet(path_2014 + 'hies_2014_consol_trimmedoutliers_groupandbase.parquet')
df_16_withbase.to_parquet(path_2016 + 'hies_2016_consol_trimmedoutliers_groupandbase.parquet')
df_16_withbase.to_parquet(path_2019 + 'hies_2019_consol_trimmedoutliers_groupandbase.parquet')

df_14_hhbasis_withbase.to_parquet(path_2014 + 'hies_2014_consol_hhbasis_trimmedoutliers_groupandbase.parquet')
df_16_hhbasis_withbase.to_parquet(path_2016 + 'hies_2016_consol_hhbasis_trimmedoutliers_groupandbase.parquet')
df_16_hhbasis_withbase.to_parquet(path_2019 + 'hies_2019_consol_hhbasis_trimmedoutliers_groupandbase.parquet')

df_14_full_withbase.to_parquet(path_2014 + 'hies_2014_consol_groupandbase.parquet')
df_16_full_withbase.to_parquet(path_2016 + 'hies_2016_consol_groupandbase.parquet')
df_16_full_withbase.to_parquet(path_2019 + 'hies_2019_consol_groupandbase.parquet')

df_14_full_hhbasis_withbase.to_parquet(path_2014 + 'hies_2014_consol_hhbasis_groupandbase.parquet')
df_16_full_hhbasis_withbase.to_parquet(path_2016 + 'hies_2016_consol_hhbasis_groupandbase.parquet')
df_16_full_hhbasis_withbase.to_parquet(path_2019 + 'hies_2019_consol_hhbasis_groupandbase.parquet')

# Delete base columns that have been transformed into categoricals
cols_base_group_transformed = \
    [
        'age',
        'income_gen_members',
        'non_working_adult_females',
        'working_adult_females',
        'child',
        'adolescent',
        'elderly',
        'birth_year'
    ]
if hhbasis_cohorts_with_hhsize:
    cols_base_group_transformed_with_hhsize = cols_base_group_transformed + ['hh_size']
elif not hhbasis_cohorts_with_hhsize:
    cols_base_group_transformed_with_hhsize = cols_base_group_transformed.copy()
for i in cols_base_group_transformed:
    del df_14[i]
    del df_16[i]
    del df_19[i]
    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_hhbasis[i]
    del df_16_hhbasis[i]
    del df_19_hhbasis[i]
    del df_14_full_hhbasis[i]
    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]

# IV.A --- Merger (group-level; 2014 - 2019)
# Observables to be grouped on
col_groups = \
    [
        'state',
        'urban',
        'education',
        'ethnicity',
        'malaysian',
        'income_gen_members_group',
        'non_working_adult_females_group',
        'working_adult_females_group',
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
if hhbasis_cohorts_with_hhsize:
    col_groups_with_hhsize = col_groups + ['hh_size_group']
elif not hhbasis_cohorts_with_hhsize:
    col_groups_with_hhsize = col_groups.copy()


# Functions
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


# The merging part
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

df_agg_mean_hhbasis, df_agg_balanced_mean_hhbasis = gen_pseudopanel(
    data1=df_14_hhbasis,
    data2=df_16_hhbasis,
    data3=df_19_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_hhbasis'
)

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
    df_agg_quantile_hhbasis, df_agg_balanced_quantile_hhbasis = gen_pseudopanel_distgroup_fixed_axis(
        data1=df_14_hhbasis,
        data2=df_16_hhbasis,
        data3=df_19_hhbasis,
        fixed_axis='gross_income',
        list_cols_cohort=col_groups_with_hhsize,
        distbounds=distbound,
        file_suffix=suffix + '_hhbasis'
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

# IV.B --------------------- Merger (group-level; 2014 - 2019; for SUBGROUP analyses) ---------------------

# Remove years
del df_14['year']
del df_16['year']
del df_19['year']

# B40
df_14_b40 = df_14_withbase[df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)].copy()
df_16_b40 = df_16_withbase[df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)].copy()
df_19_b40 = df_19_withbase[df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)].copy()

df_14_b40_hhbasis = \
    df_14_hhbasis_withbase[
        df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
        ].copy()
df_16_b40_hhbasis = \
    df_16_hhbasis_withbase[
        df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
        ].copy()
df_19_b40_hhbasis = \
    df_19_hhbasis_withbase[
        df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
        ].copy()

for i in cols_base_group_transformed:
    del df_14_b40[i]
    del df_16_b40[i]
    del df_19_b40[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_hhbasis[i]
    del df_16_b40_hhbasis[i]
    del df_19_b40_hhbasis[i]

df_agg_mean_b40, df_agg_balanced_mean_b40 = gen_pseudopanel(
    data1=df_14_b40,
    data2=df_16_b40,
    data3=df_19_b40,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40'
)
df_agg_mean_b40_hhbasis, df_agg_balanced_mean_b40_hhbasis = gen_pseudopanel(
    data1=df_14_b40_hhbasis,
    data2=df_16_b40_hhbasis,
    data3=df_19_b40_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_hhbasis'
)

# B60
df_14_b60 = df_14_withbase[df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.6)].copy()
df_16_b60 = df_16_withbase[df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.6)].copy()
df_19_b60 = df_19_withbase[df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.6)].copy()

df_14_b60_hhbasis = \
    df_14_hhbasis_withbase[
        df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.6)
        ].copy()
df_16_b60_hhbasis = \
    df_16_hhbasis_withbase[
        df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.6)
        ].copy()
df_19_b60_hhbasis = \
    df_19_hhbasis_withbase[
        df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.6)
        ].copy()

for i in cols_base_group_transformed:
    del df_14_b60[i]
    del df_16_b60[i]
    del df_19_b60[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b60_hhbasis[i]
    del df_16_b60_hhbasis[i]
    del df_19_b60_hhbasis[i]

df_agg_mean_b60, df_agg_balanced_mean_b60 = gen_pseudopanel(
    data1=df_14_b60,
    data2=df_16_b60,
    data3=df_19_b60,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b60'
)
df_agg_mean_b60_hhbasis, df_agg_balanced_mean_b60_hhbasis = gen_pseudopanel(
    data1=df_14_b60_hhbasis,
    data2=df_16_b60_hhbasis,
    data3=df_19_b60_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b60_hhbasis'
)

# B80
df_14_b80 = df_14_withbase[df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.8)].copy()
df_16_b80 = df_16_withbase[df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.8)].copy()
df_19_b80 = df_19_withbase[df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.8)].copy()

df_14_b80_hhbasis = \
    df_14_hhbasis_withbase[
        df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.8)
        ].copy()
df_16_b80_hhbasis = \
    df_16_hhbasis_withbase[
        df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.8)
        ].copy()
df_19_b80_hhbasis = \
    df_19_hhbasis_withbase[
        df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.8)
        ].copy()

for i in cols_base_group_transformed:
    del df_14_b80[i]
    del df_16_b80[i]
    del df_19_b80[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b80_hhbasis[i]
    del df_16_b80_hhbasis[i]
    del df_19_b80_hhbasis[i]

df_agg_mean_b80, df_agg_balanced_mean_b80 = gen_pseudopanel(
    data1=df_14_b80,
    data2=df_16_b80,
    data3=df_19_b80,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b80'
)

df_agg_mean_b80_hhbasis, df_agg_balanced_mean_b80_hhbasis = gen_pseudopanel(
    data1=df_14_b80_hhbasis,
    data2=df_16_b80_hhbasis,
    data3=df_19_b80_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b80_hhbasis'
)

# B40 with no children
df_14_b40_0c = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 0)
            )
    )
].copy()
df_16_b40_0c = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 0)
            )
    )
].copy()
df_19_b40_0c = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 0)
            )
    )
].copy()

df_14_b40_0c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 0)
            )
    )
].copy()
df_16_b40_0c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 0)
            )
    )
].copy()
df_19_b40_0c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            ) &
            (
                    (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 0)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_0c[i]
    del df_16_b40_0c[i]
    del df_19_b40_0c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_0c_hhbasis[i]
    del df_16_b40_0c_hhbasis[i]
    del df_19_b40_0c_hhbasis[i]

df_agg_mean_b40_0c, df_agg_balanced_mean_b40_0c = gen_pseudopanel(
    data1=df_14_b40_0c,
    data2=df_16_b40_0c,
    data3=df_19_b40_0c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_0c'
)
df_agg_mean_b40_0c_hhbasis, df_agg_balanced_mean_b40_0c_hhbasis = gen_pseudopanel(
    data1=df_14_b40_0c_hhbasis,
    data2=df_16_b40_0c_hhbasis,
    data3=df_19_b40_0c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_0c_hhbasis'
)

# B40 with only one child / adolescents
df_14_b40_1c = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_16_b40_1c = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_19_b40_1c = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()

df_14_b40_1c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_16_b40_1c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_19_b40_1c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_1c[i]
    del df_16_b40_1c[i]
    del df_19_b40_1c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_1c_hhbasis[i]
    del df_16_b40_1c_hhbasis[i]
    del df_19_b40_1c_hhbasis[i]

df_agg_mean_b40_1c, df_agg_balanced_mean_b40_1c = gen_pseudopanel(
    data1=df_14_b40_1c_hhbasis,
    data2=df_16_b40_1c,
    data3=df_19_b40_1c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_1c'
)
df_agg_mean_b40_1c_hhbasis, df_agg_balanced_mean_b40_1c_hhbasis = gen_pseudopanel(
    data1=df_14_b40_1c_hhbasis,
    data2=df_16_b40_1c_hhbasis,
    data3=df_19_b40_1c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_1c_hhbasis'
)

# B40 with at least one child / adolescents
df_14_b40_1cplus = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 0) & (df_14_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_16_b40_1cplus = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 0) & (df_16_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_19_b40_1cplus = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 0) & (df_19_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()

df_14_b40_1cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 0) & (df_14_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_16_b40_1cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 0) & (df_16_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_19_b40_1cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 0) & (df_19_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_1cplus[i]
    del df_16_b40_1cplus[i]
    del df_19_b40_1cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_1cplus_hhbasis[i]
    del df_16_b40_1cplus_hhbasis[i]
    del df_19_b40_1cplus_hhbasis[i]

df_agg_mean_b40_1cplus, df_agg_balanced_mean_b40_1cplus = gen_pseudopanel(
    data1=df_14_b40_1cplus_hhbasis,
    data2=df_16_b40_1cplus,
    data3=df_19_b40_1cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_1cplus'
)
df_agg_mean_b40_1cplus_hhbasis, df_agg_balanced_mean_b40_1cplus_hhbasis = gen_pseudopanel(
    data1=df_14_b40_1cplus_hhbasis,
    data2=df_16_b40_1cplus_hhbasis,
    data3=df_19_b40_1cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_1cplus_hhbasis'
)

# B40 with only two child / adolescents
df_14_b40_2c = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] == 2) & (df_14_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_16_b40_2c = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] == 2) & (df_16_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_19_b40_2c = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] == 2) & (df_19_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()

df_14_b40_2c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] == 2) & (df_14_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_16_b40_2c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] == 2) & (df_16_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()
df_19_b40_2c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] == 2) & (df_19_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 2)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 1)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_2c[i]
    del df_16_b40_2c[i]
    del df_19_b40_2c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_2c_hhbasis[i]
    del df_16_b40_2c_hhbasis[i]
    del df_19_b40_2c_hhbasis[i]

df_agg_mean_b40_2c, df_agg_balanced_mean_b40_2c = gen_pseudopanel(
    data1=df_14_b40_2c,
    data2=df_16_b40_2c,
    data3=df_19_b40_2c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_2c'
)
df_agg_mean_b40_2c_hhbasis, df_agg_balanced_mean_b40_2c_hhbasis = gen_pseudopanel(
    data1=df_14_b40_2c_hhbasis,
    data2=df_16_b40_2c_hhbasis,
    data3=df_19_b40_2c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_2c_hhbasis'
)

# B40 with at least two child / adolescents
df_14_b40_2cplus = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 0) & (df_14_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_16_b40_2cplus = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 0) & (df_16_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_19_b40_2cplus = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 0) & (df_19_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()

df_14_b40_2cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 0) & (df_14_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_16_b40_2cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 0) & (df_16_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()
df_19_b40_2cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 0) & (df_19_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 1)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_2cplus[i]
    del df_16_b40_2cplus[i]
    del df_19_b40_2cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_2cplus_hhbasis[i]
    del df_16_b40_2cplus_hhbasis[i]
    del df_19_b40_2cplus_hhbasis[i]

df_agg_mean_b40_2cplus, df_agg_balanced_mean_b40_2cplus = gen_pseudopanel(
    data1=df_14_b40_2cplus,
    data2=df_16_b40_2cplus,
    data3=df_19_b40_2cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_2cplus'
)
df_agg_mean_b40_2cplus_hhbasis, df_agg_balanced_mean_b40_2cplus_hhbasis = gen_pseudopanel(
    data1=df_14_b40_2cplus_hhbasis,
    data2=df_16_b40_2cplus_hhbasis,
    data3=df_19_b40_2cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_2cplus_hhbasis'
)

# B40 with only three child / adolescents
df_14_b40_3c = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] == 3) & (df_14_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 2) & (df_14_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()
df_16_b40_3c = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] == 3) & (df_16_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 2) & (df_16_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()
df_19_b40_3c = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] == 3) & (df_19_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 2) & (df_19_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()

df_14_b40_3c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] == 3) & (df_14_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 2) & (df_14_hhbasis_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()
df_16_b40_3c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] == 3) & (df_16_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 2) & (df_16_hhbasis_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()
df_19_b40_3c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] == 3) & (df_19_hhbasis_withbase['adolescent'] == 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 3)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 2) & (df_19_hhbasis_withbase['adolescent'] == 1)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 2)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_3c[i]
    del df_16_b40_3c[i]
    del df_19_b40_3c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_3c_hhbasis[i]
    del df_16_b40_3c_hhbasis[i]
    del df_19_b40_3c_hhbasis[i]

df_agg_mean_b40_3c, df_agg_balanced_mean_b40_3c = gen_pseudopanel(
    data1=df_14_b40_3c,
    data2=df_16_b40_3c,
    data3=df_19_b40_3c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_3c'
)
df_agg_mean_b40_3c_hhbasis, df_agg_balanced_mean_b40_3c_hhbasis = gen_pseudopanel(
    data1=df_14_b40_3c_hhbasis,
    data2=df_16_b40_3c_hhbasis,
    data3=df_19_b40_3c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_3c_hhbasis'
)

# B40 with more than three child / adolescents
df_14_b40_4cplus = df_14_withbase[
    (
            (
                    df_14_withbase['gross_income'] <= df_14_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_withbase['child'] >= 4) & (df_14_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 3) & (df_14_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()
df_16_b40_4cplus = df_16_withbase[
    (
            (
                    df_16_withbase['gross_income'] <= df_16_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_withbase['child'] >= 4) & (df_16_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 3) & (df_16_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()
df_19_b40_4cplus = df_19_withbase[
    (
            (
                    df_19_withbase['gross_income'] <= df_19_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_withbase['child'] >= 4) & (df_19_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 3) & (df_19_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()

df_14_b40_4cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    df_14_hhbasis_withbase['gross_income'] <= df_14_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_14_hhbasis_withbase['child'] >= 4) & (df_14_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 3) & (df_14_hhbasis_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()
df_16_b40_4cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    df_16_hhbasis_withbase['gross_income'] <= df_16_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_16_hhbasis_withbase['child'] >= 4) & (df_16_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 3) & (df_16_hhbasis_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()
df_19_b40_4cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    df_19_hhbasis_withbase['gross_income'] <= df_19_hhbasis_withbase['gross_income'].quantile(q=0.4)
            )
            &
            (
                    (
                            (df_19_hhbasis_withbase['child'] >= 4) & (df_19_hhbasis_withbase['adolescent'] >= 0)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 4)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 3) & (df_19_hhbasis_withbase['adolescent'] >= 1)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 3)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 2)
                    )
                    |
                    (
                            (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 2)
                    )
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_b40_4cplus[i]
    del df_16_b40_4cplus[i]
    del df_19_b40_4cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_b40_4cplus_hhbasis[i]
    del df_16_b40_4cplus_hhbasis[i]
    del df_19_b40_4cplus_hhbasis[i]

df_agg_mean_b40_4cplus, df_agg_balanced_mean_b40_4cplus = gen_pseudopanel(
    data1=df_14_b40_4cplus,
    data2=df_16_b40_4cplus,
    data3=df_19_b40_4cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_4cplus'
)
df_agg_mean_b40_4cplus_hhbasis, df_agg_balanced_mean_b40_4cplus_hhbasis = gen_pseudopanel(
    data1=df_14_b40_4cplus_hhbasis,
    data2=df_16_b40_4cplus_hhbasis,
    data3=df_19_b40_4cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_b40_4cplus_hhbasis'
)

# No children
df_14_0c = df_14_withbase[
    (
            (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 0)
    )
].copy()
df_16_0c = df_16_withbase[
    (
            (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 0)
    )
].copy()
df_19_0c = df_19_withbase[
    (
            (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 0)
    )
].copy()

df_14_0c_hhbasis = df_14_hhbasis_withbase[
    (
            (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 0)
    )
].copy()
df_16_0c_hhbasis = df_16_hhbasis_withbase[
    (
            (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 0)
    )
].copy()
df_19_0c_hhbasis = df_19_hhbasis_withbase[
    (
            (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 0)
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_0c[i]
    del df_16_0c[i]
    del df_19_0c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_0c_hhbasis[i]
    del df_16_0c_hhbasis[i]
    del df_19_0c_hhbasis[i]

df_agg_mean_0c, df_agg_balanced_mean_0c = gen_pseudopanel(
    data1=df_14_0c,
    data2=df_16_0c,
    data3=df_19_0c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_0c'
)
df_agg_mean_0c_hhbasis, df_agg_balanced_mean_0c_hhbasis = gen_pseudopanel(
    data1=df_14_0c_hhbasis,
    data2=df_16_0c_hhbasis,
    data3=df_19_0c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_0c_hhbasis'
)

# Only one child / adolescents
df_14_1c = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 1)
            )
    )
].copy()
df_16_1c = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 1)
            )
    )
].copy()
df_19_1c = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 1)
            )
    )
].copy()

df_14_1c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()
df_16_1c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()
df_19_1c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_1c[i]
    del df_16_1c[i]
    del df_19_1c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_1c_hhbasis[i]
    del df_16_1c_hhbasis[i]
    del df_19_1c_hhbasis[i]

df_agg_mean_1c, df_agg_balanced_mean_1c = gen_pseudopanel(
    data1=df_14_1c,
    data2=df_16_1c,
    data3=df_19_1c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_1c'
)
df_agg_mean_1c_hhbasis, df_agg_balanced_mean_1c_hhbasis = gen_pseudopanel(
    data1=df_14_1c_hhbasis,
    data2=df_16_1c_hhbasis,
    data3=df_19_1c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_1c_hhbasis'
)

# At least one child / adolescents
df_14_1cplus = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_withbase['child'] >= 0) & (df_14_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_16_1cplus = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_withbase['child'] >= 0) & (df_16_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_19_1cplus = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_withbase['child'] >= 0) & (df_19_withbase['adolescent'] >= 1)
            )
    )
].copy()

df_14_1cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 0) & (df_14_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_16_1cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 0) & (df_16_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_19_1cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 0) & (df_19_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_1cplus[i]
    del df_16_1cplus[i]
    del df_19_1cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_1cplus_hhbasis[i]
    del df_16_1cplus_hhbasis[i]
    del df_19_1cplus_hhbasis[i]

df_agg_mean_1cplus, df_agg_balanced_mean_1cplus = gen_pseudopanel(
    data1=df_14_1cplus,
    data2=df_16_1cplus,
    data3=df_19_1cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_1cplus'
)
df_agg_mean_1cplus_hhbasis, df_agg_balanced_mean_1cplus_hhbasis = gen_pseudopanel(
    data1=df_14_1cplus_hhbasis,
    data2=df_16_1cplus_hhbasis,
    data3=df_19_1cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_1cplus_hhbasis'
)


# Only two child / adolescents
df_14_2c = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] == 2) & (df_14_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 1)
            )
    )
].copy()
df_16_2c = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] == 2) & (df_16_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 1)
            )
    )
].copy()
df_19_2c = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] == 2) & (df_19_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 1)
            )
    )
].copy()

df_14_2c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] == 2) & (df_14_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()
df_16_2c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] == 2) & (df_16_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()
df_19_2c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] == 2) & (df_19_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 2)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 1)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_2c[i]
    del df_16_2c[i]
    del df_19_2c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_2c_hhbasis[i]
    del df_16_2c_hhbasis[i]
    del df_19_2c_hhbasis[i]

df_agg_mean_2c, df_agg_balanced_mean_2c = gen_pseudopanel(
    data1=df_14_2c,
    data2=df_16_2c,
    data3=df_19_2c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_2c'
)
df_agg_mean_2c_hhbasis, df_agg_balanced_mean_2c_hhbasis = gen_pseudopanel(
    data1=df_14_2c_hhbasis,
    data2=df_16_2c_hhbasis,
    data3=df_19_2c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_2c_hhbasis'
)

# At least two child / adolescents
df_14_2cplus = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_withbase['child'] >= 0) & (df_14_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_16_2cplus = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_withbase['child'] >= 0) & (df_16_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_19_2cplus = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_withbase['child'] >= 0) & (df_19_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 1)
            )
    )
].copy()

df_14_2cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 0) & (df_14_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_16_2cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 0) & (df_16_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()
df_19_2cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 0) & (df_19_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 1)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_2cplus[i]
    del df_16_2cplus[i]
    del df_19_2cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_2cplus_hhbasis[i]
    del df_16_2cplus_hhbasis[i]
    del df_19_2cplus_hhbasis[i]

df_agg_mean_2cplus, df_agg_balanced_mean_2cplus = gen_pseudopanel(
    data1=df_14_2cplus,
    data2=df_16_2cplus,
    data3=df_19_2cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_2cplus'
)
df_agg_mean_2cplus_hhbasis, df_agg_balanced_mean_2cplus_hhbasis = gen_pseudopanel(
    data1=df_14_2cplus_hhbasis,
    data2=df_16_2cplus_hhbasis,
    data3=df_19_2cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_2cplus_hhbasis'
)

# Only three child / adolescents
df_14_3c = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] == 3) & (df_14_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_14_withbase['child'] == 2) & (df_14_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_14_withbase['child'] == 1) & (df_14_withbase['adolescent'] == 2)
            )
    )
].copy()
df_16_3c = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] == 3) & (df_16_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_16_withbase['child'] == 2) & (df_16_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_16_withbase['child'] == 1) & (df_16_withbase['adolescent'] == 2)
            )
    )
].copy()
df_19_3c = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] == 3) & (df_19_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_19_withbase['child'] == 2) & (df_19_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_19_withbase['child'] == 1) & (df_19_withbase['adolescent'] == 2)
            )
    )
].copy()

df_14_3c_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] == 3) & (df_14_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 2) & (df_14_hhbasis_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 1) & (df_14_hhbasis_withbase['adolescent'] == 2)
            )
    )
].copy()
df_16_3c_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] == 3) & (df_16_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 2) & (df_16_hhbasis_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 1) & (df_16_hhbasis_withbase['adolescent'] == 2)
            )
    )
].copy()
df_19_3c_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] == 3) & (df_19_hhbasis_withbase['adolescent'] == 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 3)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 2) & (df_19_hhbasis_withbase['adolescent'] == 1)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 1) & (df_19_hhbasis_withbase['adolescent'] == 2)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_3c[i]
    del df_16_3c[i]
    del df_19_3c[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_3c_hhbasis[i]
    del df_16_3c_hhbasis[i]
    del df_19_3c_hhbasis[i]

df_agg_mean_3c, df_agg_balanced_mean_3c = gen_pseudopanel(
    data1=df_14_3c,
    data2=df_16_3c,
    data3=df_19_3c,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_3c'
)
df_agg_mean_3c_hhbasis, df_agg_balanced_mean_3c_hhbasis = gen_pseudopanel(
    data1=df_14_3c_hhbasis,
    data2=df_16_3c_hhbasis,
    data3=df_19_3c_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_3c_hhbasis'
)

# More than three child / adolescents
df_14_4cplus = df_14_withbase[
    (
            (
                    (df_14_withbase['child'] >= 4) & (df_14_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_withbase['child'] == 0) & (df_14_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_14_withbase['child'] >= 3) & (df_14_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_14_withbase['child'] >= 1) & (df_14_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_14_withbase['child'] >= 2) & (df_14_withbase['adolescent'] >= 2)
            )
    )
].copy()
df_16_4cplus = df_16_withbase[
    (
            (
                    (df_16_withbase['child'] >= 4) & (df_16_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_withbase['child'] == 0) & (df_16_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_16_withbase['child'] >= 3) & (df_16_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_16_withbase['child'] >= 1) & (df_16_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_16_withbase['child'] >= 2) & (df_16_withbase['adolescent'] >= 2)
            )
    )
].copy()
df_19_4cplus = df_19_withbase[
    (
            (
                    (df_19_withbase['child'] >= 4) & (df_19_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_withbase['child'] == 0) & (df_19_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_19_withbase['child'] >= 3) & (df_19_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_19_withbase['child'] >= 1) & (df_19_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_19_withbase['child'] >= 2) & (df_19_withbase['adolescent'] >= 2)
            )
    )
].copy()

df_14_4cplus_hhbasis = df_14_hhbasis_withbase[
    (
            (
                    (df_14_hhbasis_withbase['child'] >= 4) & (df_14_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] == 0) & (df_14_hhbasis_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 3) & (df_14_hhbasis_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 1) & (df_14_hhbasis_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_14_hhbasis_withbase['child'] >= 2) & (df_14_hhbasis_withbase['adolescent'] >= 2)
            )
    )
].copy()
df_16_4cplus_hhbasis = df_16_hhbasis_withbase[
    (
            (
                    (df_16_hhbasis_withbase['child'] >= 4) & (df_16_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] == 0) & (df_16_hhbasis_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 3) & (df_16_hhbasis_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 1) & (df_16_hhbasis_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_16_hhbasis_withbase['child'] >= 2) & (df_16_hhbasis_withbase['adolescent'] >= 2)
            )
    )
].copy()
df_19_4cplus_hhbasis = df_19_hhbasis_withbase[
    (
            (
                    (df_19_hhbasis_withbase['child'] >= 4) & (df_19_hhbasis_withbase['adolescent'] >= 0)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] == 0) & (df_19_hhbasis_withbase['adolescent'] == 4)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 3) & (df_19_hhbasis_withbase['adolescent'] >= 1)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 1) & (df_19_hhbasis_withbase['adolescent'] >= 3)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 2)
            )
            |
            (
                    (df_19_hhbasis_withbase['child'] >= 2) & (df_19_hhbasis_withbase['adolescent'] >= 2)
            )
    )
].copy()

for i in cols_base_group_transformed:
    del df_14_4cplus[i]
    del df_16_4cplus[i]
    del df_19_4cplus[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_4cplus_hhbasis[i]
    del df_16_4cplus_hhbasis[i]
    del df_19_4cplus_hhbasis[i]

df_agg_mean_4cplus, df_agg_balanced_mean_4cplus = gen_pseudopanel(
    data1=df_14_4cplus,
    data2=df_16_4cplus,
    data3=df_19_4cplus,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_4cplus'
)
df_agg_mean_4cplus_hhbasis, df_agg_balanced_mean_4cplus_hhbasis = gen_pseudopanel(
    data1=df_14_4cplus_hhbasis,
    data2=df_16_4cplus_hhbasis,
    data3=df_19_4cplus_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_4cplus_hhbasis'
)

# All households whose head is a working-age adult
df_14_adults = df_14_withbase[(df_14_withbase['age'] >= 18) & (df_14_withbase['age'] < 60)].copy()
df_16_adults = df_16_withbase[(df_16_withbase['age'] >= 18) & (df_16_withbase['age'] < 60)].copy()
df_19_adults = df_19_withbase[(df_19_withbase['age'] >= 18) & (df_19_withbase['age'] < 60)].copy()

df_14_adults_hhbasis = \
    df_14_hhbasis_withbase[
        (df_14_hhbasis_withbase['age'] >= 18) & (df_14_hhbasis_withbase['age'] < 60)
        ].copy()
df_16_adults_hhbasis = \
    df_16_hhbasis_withbase[
        (df_16_hhbasis_withbase['age'] >= 18) & (df_16_hhbasis_withbase['age'] < 60)
        ].copy()
df_19_adults_hhbasis = \
    df_19_hhbasis_withbase[
        (df_19_hhbasis_withbase['age'] >= 18) & (df_19_hhbasis_withbase['age'] < 60)
        ].copy()

for i in cols_base_group_transformed:
    del df_14_adults[i]
    del df_16_adults[i]
    del df_19_adults[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_adults_hhbasis[i]
    del df_16_adults_hhbasis[i]
    del df_19_adults_hhbasis[i]

df_agg_mean_adults, df_agg_balanced_mean_adults = gen_pseudopanel(
    data1=df_14_adults,
    data2=df_16_adults,
    data3=df_19_adults,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_adults'
)
df_agg_mean_adults_hhbasis, df_agg_balanced_mean_adults_hhbasis = gen_pseudopanel(
    data1=df_14_adults_hhbasis,
    data2=df_16_adults_hhbasis,
    data3=df_19_adults_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_adults_hhbasis'
)

# All elderly
df_14_elderly = df_14_withbase[df_14_withbase['age'] >= 60].copy()
df_16_elderly = df_16_withbase[df_16_withbase['age'] >= 60].copy()
df_19_elderly = df_19_withbase[df_19_withbase['age'] >= 60].copy()

df_14_elderly_hhbasis = \
    df_14_hhbasis_withbase[
        df_14_hhbasis_withbase['age'] >= 60
        ].copy()
df_16_elderly_hhbasis = \
    df_16_hhbasis_withbase[
        df_16_hhbasis_withbase['age'] >= 60
        ].copy()
df_19_elderly_hhbasis = \
    df_19_hhbasis_withbase[
        df_19_hhbasis_withbase['age'] >= 60
        ].copy()

for i in cols_base_group_transformed:
    del df_14_elderly[i]
    del df_16_elderly[i]
    del df_19_elderly[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_elderly_hhbasis[i]
    del df_16_elderly_hhbasis[i]
    del df_19_elderly_hhbasis[i]

df_agg_mean_elderly, df_agg_balanced_mean_elderly = gen_pseudopanel(
    data1=df_14_elderly,
    data2=df_16_elderly,
    data3=df_19_elderly,
    list_cols_cohort=col_groups,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_elderly'
)
df_agg_mean_elderly_hhbasis, df_agg_balanced_mean_elderly_hhbasis = gen_pseudopanel(
    data1=df_14_elderly_hhbasis,
    data2=df_16_elderly_hhbasis,
    data3=df_19_elderly_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix='mean_elderly_hhbasis'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_consol_group: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
