import pandas as pd
import numpy as np
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols, heatmap, pil_img2pdf, boxplot_time
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
hhbasis_descriptive = ast.literal_eval(os.getenv('HHBASIS_DESCRIPTIVE'))
equivalised_descriptive = ast.literal_eval(os.getenv('EQUIVALISED_DESCRIPTIVE'))
if hhbasis_descriptive:
    input_suffix = '_hhbasis'
    output_suffix = '_hhbasis'
    chart_suffix = ' (Total HH)'
if equivalised_descriptive:
    input_suffix = '_equivalised'
    output_suffix = '_equivalised'
    chart_suffix = ' (Equivalised)'
if not hhbasis_descriptive and not equivalised_descriptive:
    input_suffix = ''
    output_suffix = '_capita'
    chart_suffix = ' (Per Capita)'

# I --- Load data
df = pd.read_parquet(
    path_data + 'hies_consol_ind_full' + output_suffix + '.parquet'
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
list_cat_outcomes = ['hh_size_group'] + list_groups
# Define continuous outcome variables
list_outcomes = ['gross_income'] + \
                ['gross_transfers'] + \
                ['cons_01_13', 'cons_01_12'] + \
                ['cons_01', 'cons_04', 'cons_06', 'cons_07', 'cons_10'] + \
                ['cons_0722_fuel', 'cons_07_ex_bigticket']
# F&B, util, healthcare, transport & fuels, education
# ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']
# Define list of quantiles
list_quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                  0.95]
list_quantiles_nice = ['5p', '10p', '15p', '20p', '25p', '30p', '35p', '40p', '45p', '50p', '55p', '60p', '65p', '70p',
                       '75p', '80p', '85p', '90p', '95p']
# Create new columns for consumption category as % of income
list_all_cons = ['cons_01_13', 'cons_01_12'] + \
                ['cons_0' + str(i) for i in range(1, 10)] + \
                ['cons_1' + str(i) for i in range(0, 4)] + \
                ['cons_0722_fuel', 'cons_07_ex_bigticket']
list_all_cons_incshare = [i + '_incshare' for i in list_all_cons]
for i, j in zip(list_all_cons, list_all_cons_incshare):
    df[j] = 100 * df[i] / df['gross_income']


# Generate income groups
def gen_gross_income_group(data, aggregation):
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


gen_gross_income_group(data=df, aggregation=1)


# Generate household size groups
def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 8, 'hh_size_group'] = '8+'


gen_hh_size_group(data=df)


# III --- The analysis
# Functions
def capybara(data, y_col, x_col, t_col, y_quantiles):
    # prelims
    d = data.copy()
    # compute quantiles by time (t), and by group (x)
    round = 1
    for quantile in y_quantiles:
        tab = d.groupby([x_col, t_col])[y_col].quantile(q=quantile).reset_index()
        tab['quantile'] = quantile
        if round == 1:
            tab_consol = tab.copy()
        elif round > 1:
            tab_consol = pd.concat([tab_consol, tab], axis=0)
        round += 1
    # compute growth
    tab_consol_growth = tab_consol.copy()
    tab_consol_growth = tab_consol_growth.sort_values(by=['quantile', x_col, t_col], ascending=[True, True, True])
    tab_consol_growth[y_col] = 100 * (
            (tab_consol_growth[y_col] / tab_consol_growth.groupby(['quantile', x_col])[y_col].shift(1)) - 1
    )
    tab_consol_growth = tab_consol_growth.dropna(axis=0)
    # compute change
    tab_consol_change = tab_consol.copy()
    tab_consol_change = tab_consol_change.sort_values(by=['quantile', x_col, t_col], ascending=[True, True, True])
    tab_consol_change[y_col] = tab_consol_change[y_col] - tab_consol_change.groupby(['quantile', x_col])[y_col].shift(1)
    tab_consol_change = tab_consol_change.dropna(axis=0)
    # output
    return tab_consol, tab_consol_growth, tab_consol_change


# Generate level, change, and growth by income quantiles
tab_consol, tab_consol_growth, tab_consol_change = capybara(
    data=df,
    y_col='gross_income',
    x_col='gross_income_group',
    t_col='year',
    y_quantiles=list_quantiles
)
# Convert values into 2dp max
tab_consol['gross_income'] = tab_consol['gross_income'].round(2)
tab_consol_growth['gross_income'] = tab_consol_growth['gross_income'].round(2)
tab_consol_change['gross_income'] = tab_consol_change['gross_income'].round(2)
# Combine tables for growth and change
tab_consol_change = tab_consol_change.rename(columns={'gross_income': 'change'})
tab_consol_growth = tab_consol_growth.rename(columns={'gross_income': 'growth'})
tab_consol_growth_change = tab_consol_growth.merge(
    tab_consol_change,
    how='left',
    on=['gross_income_group', 'year', 'quantile']
)
tab_consol_growth_change = tab_consol_growth_change[['gross_income_group', 'year', 'quantile', 'growth', 'change']]
list_years = [2016, 2019]
list_years_nice = ['2014-16', '2016-19']
list_files_tab = []
for year, year_nice in zip(list_years, list_years_nice):
    for quantile, quantile_nice in tqdm(zip(list_quantiles, list_quantiles_nice)):
        # restrict year and quantile
        d = tab_consol_growth_change[((tab_consol_growth_change['quantile'] == quantile) &
                                      (tab_consol_growth_change['year'] == year))].copy()
        # nice year values
        d['year'] = year_nice
        # set index
        d = d.set_index('gross_income_group')
        # generate tables
        dfi.export(d,
                   './output/tab_inc_growth_change_' + 'gross_income' + '_' + str(year) + '_' + quantile_nice + output_suffix + '.png')
        list_files_tab = list_files_tab + \
                         ['./output/tab_inc_growth_change_' + 'gross_income' + '_' + str(year) + '_' + quantile_nice + output_suffix]
# Save as single pdf
pil_img2pdf(
    list_images=list_files_tab,
    extension='png',
    pdf_name='./output/tab_inc_growth_change_' + 'gross_income' + '_' + 'year_quantile' + output_suffix
)
telsendfiles(
    conf=tel_config,
    path='./output/tab_inc_growth_change_' + 'gross_income' + '_' + 'year_quantile' + output_suffix + '.pdf',
    cap='tab_inc_growth_change_' + 'gross_income' + '_' + 'year_quantile' + output_suffix
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_inc_growth_change: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
