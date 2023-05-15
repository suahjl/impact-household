# Select descriptive stats (only income and consumption) across total HH, equivalised, and per capita basis using HH income groups

import pandas as pd
import numpy as np
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    fe_reg, re_reg, reg_ols, \
    heatmap, pil_img2pdf, boxplot_time, heatmap_layered
import matplotlib.pyplot as plt
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
use_spending_income_ratio = ast.literal_eval(os.getenv('USE_SPENDING_INCOME_RATIO'))

# I --- Load data
df = pd.read_parquet(
    path_data + 'hies_consol_ind_full' + '_hhbasis' + '.parquet'
)  # CHECK: include / exclude outliers and on hhbasis

# II --- Pre-analysis prep
# Malaysian only
df = df[df['malaysian'] == 1]
# Redefine year
df = df.rename(columns={'_time': 'year'})
# Define observables / cohort variables
list_groups = \
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
# Define categorical outcome variables to be sliced and spliced only by income groups
list_cat_outcomes = ['hh_size_group'] + list_groups  # only available for total household basis
# Define continuous outcome variables
list_outcomes = ['gross_income'] + \
                ['gross_transfers'] + \
                ['cons_01_13', 'cons_01_12'] + \
                ['cons_01', 'cons_04', 'cons_06', 'cons_07', 'cons_10'] + \
                ['cons_0722_fuel', 'cons_07_ex_bigticket']
# F&B, util, healthcare, transport & fuels, education
# ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']

# Convert consumption into % of income
list_all_cons = ['cons_01_13', 'cons_01_12'] + \
                ['cons_0' + str(i) for i in range(1, 10)] + \
                ['cons_1' + str(i) for i in range(0, 4)] + \
                ['cons_0722_fuel', 'cons_07_ex_bigticket']
if use_spending_income_ratio:
    for cons in list_all_cons:
        df[cons] = 100 * df[cons] / df['gross_income']


# III.0 --- Define function


def capybara_tiny(data, y_col, x_col):
    # Prelims
    d = data.copy()
    # Compute median and means for y (y must be continuous, and x must be categorical)
    tab_median = pd.DataFrame(d.groupby(x_col)[y_col].quantile(q=0.5))
    tab_mean = pd.DataFrame(d.groupby(x_col)[y_col].mean())
    tab_min = pd.DataFrame(d.groupby(x_col)[y_col].min())
    tab_max = pd.DataFrame(d.groupby(x_col)[y_col].max())
    # Harmonise format
    tab_median = tab_median.astype('float')
    tab_mean = tab_mean.astype('float')
    tab_min = tab_min.astype('float')
    tab_max = tab_max.astype('float')
    # Output
    return tab_median, tab_mean, tab_min, tab_max


def gen_gross_income_group(data, aggregation):
    if aggregation == -1:
        for t in tqdm(list(data['year'].unique())):
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.99)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = 'X_100p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.95)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.99)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '95p_99p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.9)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.95)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '90p_95p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.9)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '80p_90p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.7)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '70p_80p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.7)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '60p_70p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.5)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '50p_60p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.5)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '40p_50p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.3)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '30p_40p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.3)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '20_30p'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.1)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '10p_20p'
            data.loc[
                (
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.1)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '0p_10p'
    if aggregation == 0:
        for t in tqdm(list(data['year'].unique())):
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.99)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '5_t1'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.99)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '4_t19'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '3_m20+'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '2_m20-'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '1_b20+'
            data.loc[
                (
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '0_b20-'
    if aggregation == 1:
        for t in tqdm(list(data['year'].unique())):
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '4_t20'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '3_m20+'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.6)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '2_m20-'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '1_b20+'
            data.loc[
                (
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.2)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '0_b20-'
    elif aggregation == 2:
        for t in tqdm(list(data['year'].unique())):
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '2_t20'
            data.loc[
                (
                        (
                                data['gross_income'] >= data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.8)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '1_m40'
            data.loc[
                (
                        (
                                data['gross_income'] < data.loc[data['year'] == t, 'gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['year'] == t
                        )
                ),
                'gross_income_group'
            ] = '0_b40'


def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 8, 'hh_size_group'] = '8+'


def gen_income_allbasis(data):
    for col in ['gross_income', 'gross_transfers']:
        data[col + '_equivalised'] = data[col] / (data['hh_size'] ** (1 / 2))
        data[col + '_capita'] = data[col] / data['hh_size']


# III.0 --- Generate income groups
# Define income group buckets
gen_gross_income_group(data=df, aggregation=-1)  # check aggregation choice

# III.0 --- Generate income per capita and equivalised income
gen_income_allbasis(data=df)

# III.0 --- Generate HH size buckets
# Define hh size group buckets
gen_hh_size_group(data=df)

# III.A --- Stratify income (of all 3 basis) by the same def of HH income group
base_outcomes = ['gross_income', 'gross_transfers']
list_heattables_median_names = []
list_heattables_mean_names = []
list_heattables_min_names = []
list_heattables_max_names = []
list_heattables_allstats_names = []
for outcome in tqdm(base_outcomes):
    for t in [2019, 2016, 2014]:
        d = df.copy()  # deep copy
        d = d[d['year'] == t]
        heattable_median_name = 'output/tab_median_' + outcome + '_allbasis_fixedincgroups_' + str(t)
        heattable_mean_name = 'output/tab_mean_' + outcome + '_allbasis_fixedincgroups_' + str(t)
        heattable_min_name = 'output/tab_min_' + outcome + '_allbasis_fixedincgroups_' + str(t)
        heattable_max_name = 'output/tab_max_' + outcome + '_allbasis_fixedincgroups_' + str(t)
        heattable_allstats_name = 'output/tab_allstats_' + outcome + '_allbasis_fixedincgroups_' + str(t)
        list_heattables_median_names = list_heattables_median_names + [heattable_median_name]  # separate file
        list_heattables_mean_names = list_heattables_mean_names + [heattable_mean_name]  # separate file
        list_heattables_min_names = list_heattables_min_names + [heattable_min_name]  # separate file
        list_heattables_max_names = list_heattables_max_names + [heattable_max_name]  # separate file
        list_heattables_allstats_names = list_heattables_allstats_names + [heattable_allstats_name]  # separate file
        round = 1
        for basis, basis_nice in zip(['', '_equivalised', '_capita'], ['Total HH', 'Equivalised', 'Per Capita']):
            tab_median, tab_mean, tab_min, tab_max = capybara_tiny(
                data=d,  # uses year-specific data frame
                x_col='gross_income_group',  # this is defined on total hh basis
                y_col=outcome + basis,
            )
            tab_median = tab_median.rename(columns={outcome + basis: basis_nice})
            tab_mean = tab_mean.rename(columns={outcome + basis: basis_nice})
            tab_min = tab_min.rename(columns={outcome + basis: basis_nice})
            tab_max = tab_max.rename(columns={outcome + basis: basis_nice})
            if round == 1:
                tab_median_consol = tab_median.copy()
                tab_mean_consol = tab_mean.copy()
                tab_min_consol = tab_min.copy()
                tab_max_consol = tab_max.copy()
            elif round > 1:
                tab_median_consol = pd.concat([tab_median_consol, tab_median], axis=1)  # left-right
                tab_mean_consol = pd.concat([tab_mean_consol, tab_mean], axis=1)  # left-right
                tab_min_consol = pd.concat([tab_min_consol, tab_min], axis=1)  # left-right
                tab_max_consol = pd.concat([tab_max_consol, tab_max], axis=1)  # left-right
            round += 1
        tab_median_consol = tab_median_consol.transpose()
        tab_mean_consol = tab_mean_consol.transpose()
        tab_min_consol = tab_min_consol.transpose()
        tab_max_consol = tab_max_consol.transpose()
        fig_tab_median_consol = heatmap(
            input=tab_median_consol,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_median_name + '.png',
            title='Medians of HH, Equivalised, and Per Capita ' + outcome + ' for year ' + str(t),
            lb=tab_median_consol.min().max(),
            ub=tab_median_consol.max().max(),
            format='.0f'
        )
        fig_tab_mean_consol = heatmap(
            input=tab_mean_consol,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_mean_name + '.png',
            title='Means of HH, Equivalised, and Per Capita ' + outcome + ' for year ' + str(t),
            lb=tab_mean_consol.min().max(),
            ub=tab_mean_consol.max().max(),
            format='.0f'
        )
        fig_tab_min_consol = heatmap(
            input=tab_min_consol,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_min_name + '.png',
            title='Min of HH, Equivalised, and Per Capita ' + outcome + ' for year ' + str(t),
            lb=tab_min_consol.min().max(),
            ub=tab_min_consol.max().max(),
            format='.0f'
        )
        fig_tab_max_consol = heatmap(
            input=tab_max_consol,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_max_name + '.png',
            title='Max of HH, Equivalised, and Per Capita ' + outcome + ' for year ' + str(t),
            lb=tab_max_consol.min().max(),
            ub=tab_max_consol.max().max(),
            format='.0f'
        )
        # Generate allstats version
        tab_allstats_consol = tab_min_consol.round(0).astype('int').astype('str') + \
                              '\n' + \
                              ' to ' + \
                              '\n' + \
                              tab_max_consol.round(0).astype('int').astype('str') + \
                              '\n' + \
                              '(' + tab_median_consol.round(0).astype('int').astype('str') + \
                              ')'
        fig_tab_allstats_consol = heatmap_layered(
            actual_input=tab_median_consol,
            disp_input=tab_allstats_consol,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_allstats_name + '.png',
            title='Range (Median) of HH, Equivalised, and Per Capita ' + outcome + ' for year ' + str(t),
            lb=tab_median_consol.min().max(),
            ub=tab_median_consol.max().max(),
            format='s'
        )

pil_img2pdf(
    list_images=list_heattables_median_names,
    extension='png',
    pdf_name='output/tab_median_inc_allbasis_fixedincgroups_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_median_inc_allbasis_fixedincgroups_years.pdf',
    cap='tab_median_inc_allbasis_fixedincgroups_years'
)

pil_img2pdf(
    list_images=list_heattables_mean_names,
    extension='png',
    pdf_name='output/tab_mean_inc_allbasis_fixedincgroups_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_mean_inc_allbasis_fixedincgroups_years.pdf',
    cap='tab_mean_inc_allbasis_fixedincgroups_years'
)

pil_img2pdf(
    list_images=list_heattables_min_names,
    extension='png',
    pdf_name='output/tab_min_inc_allbasis_fixedincgroups_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_min_inc_allbasis_fixedincgroups_years.pdf',
    cap='tab_min_inc_allbasis_fixedincgroups_years'
)

pil_img2pdf(
    list_images=list_heattables_max_names,
    extension='png',
    pdf_name='output/tab_max_inc_allbasis_fixedincgroups_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_max_inc_allbasis_fixedincgroups_years.pdf',
    cap='tab_max_inc_allbasis_fixedincgroups_years'
)

pil_img2pdf(
    list_images=list_heattables_allstats_names,
    extension='png',
    pdf_name='output/tab_allstats_inc_allbasis_fixedincgroups_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_allstats_inc_allbasis_fixedincgroups_years.pdf',
    cap='tab_allstats_inc_allbasis_fixedincgroups_years'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_fixedincgroups: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
