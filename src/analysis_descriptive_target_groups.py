# Special analysis (ad hoc, highly specific ones)
# Descriptive stats

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
path_2019 = './data/hies_2019/'

# I --- Load data
df = pd.read_parquet(
    path_2019 + 'hies_2019_consol_hhbasis.parquet')  # CHECK: include / exclude outliers and on hhbasis

# II --- Pre-analysis prep
# Malaysian only
df = df[df['malaysian'] == 1]
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
list_cat_outcomes = ['hh_size'] + list_groups
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
        data.loc[
            (
                    data['gross_income'] >= data['gross_income'].quantile(q=0.8)
            ),
            'gross_income_group'
        ] = '4_t20'
        data.loc[
            (
                    (
                            data['gross_income'] >= data['gross_income'].quantile(q=0.6)
                    ) &
                    (
                            data['gross_income'] < data['gross_income'].quantile(q=0.8)
                    )
            ),
            'gross_income_group'
        ] = '3_m20+'
        data.loc[
            (
                    (
                            data['gross_income'] >= data['gross_income'].quantile(q=0.4)
                    ) &
                    (
                            data['gross_income'] < data['gross_income'].quantile(q=0.6)
                    )
            ),
            'gross_income_group'
        ] = '2_m20-'
        data.loc[
            (
                    (
                            data['gross_income'] >= data['gross_income'].quantile(q=0.2)
                    ) &
                    (
                            data['gross_income'] < data['gross_income'].quantile(q=0.4)
                    )
            ),
            'gross_income_group'
        ] = '1_b20+'
        data.loc[
            (
                    data['gross_income'] < data['gross_income'].quantile(q=0.2)
            ),
            'gross_income_group'
        ] = '0_b20-'
    elif aggregation == 2:
        for t in tqdm(list(data['year'].unique())):
            data.loc[
                (
                        data['gross_income'] >= data['gross_income'].quantile(q=0.8)
                )
                ,
                'gross_income_group'
            ] = '2_t20'
            data.loc[
                (
                        (
                                data['gross_income'] >= data['gross_income'].quantile(q=0.4)
                        ) &
                        (
                                data['gross_income'] < data['gross_income'].quantile(q=0.8)
                        )
                ),
                'gross_income_group'
            ] = '1_m40'
            data.loc[
                (
                        data['gross_income'] < data['gross_income'].quantile(q=0.4)
                )
                ,
                'gross_income_group'
            ] = '0_b40'


gen_gross_income_group(data=df, aggregation=1)


# Generate household size groups
def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 10, 'hh_size_group'] = '10+'


gen_hh_size_group(data=df)

# III --- The analysis
# Restrict to 2019
# Total HH
n_total = len(df)
# Share of B40 households with at least 1 kid
n_b40_at_least_one_kid = df[((~(df['child_group'] == '0') |
                              ~(df['adolescent_group'] == '0')) &
                             ((df['gross_income_group'] == '0_b20-') |
                              (df['gross_income_group'] == '1_b20+')))].count().reset_index(drop=True)[0]
perc_b40_at_least_one_kid = 100 * n_b40_at_least_one_kid / n_total
# Share of B40 households with 0 kids
n_b40_zero_kid = df[(((df['child_group'] == '0') &
                      (df['adolescent_group'] == '0')) &
                     ((df['gross_income_group'] == '0_b20-') |
                      (df['gross_income_group'] == '1_b20+')))].count().reset_index(drop=True)[0]
perc_b40_zero_kid = 100 * n_b40_zero_kid / n_total
# Output
print('B40 with at least 1 kid: ' + str(perc_b40_at_least_one_kid) + '%' + '\n' +
      'B40 with 0 kid: ' + str(perc_b40_zero_kid) + '%')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_target_groups: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
