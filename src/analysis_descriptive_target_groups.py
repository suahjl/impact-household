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
path_output= './output/'
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
    path_2019 + 'hies_2019_consol' + input_suffix + '.parquet')  # CHECK: include / exclude outliers and on hhbasis

# II --- Pre-analysis prep
# Malaysian only
df = df[df['malaysian'] == 1]
# Define observables / cohort variables
list_groups = \
    [
        'working_age2',
        'elderly2',
        'kid'
    ]
# Define continuous outcome variables
list_outcomes = ['gross_income']
# Trim columns
df = df[list_outcomes + list_groups]
# Rename columns
df = df.rename(
    columns={
        'working_age2': 'Working Age',
        'elderly2': 'Elderly',
        'kid': 'Child'
    }
)
list_groups = ['Working Age', 'Elderly', 'Child']

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
df['gross_income_group'] = df['gross_income_group'].replace(
    {
        '4_t20': '4_T20',
        '3_m20+': '3_M20+',
        '2_m20-': '2_M20-',
        '1_b20+': '1_B20+',
        '0_b20-': '0_B20-',
    }
)

# III --- The analysis
# Number of citizens (census)
n_mys = 29756315
# Number of households
# n_hh_mys = 8234644

# Tabulate sum (HH --> ind) of age group by income group
tab = df.groupby('gross_income_group')[list_groups].sum()
tabperc = 100 * tab / tab.sum().sum()
tab_scaled = (tabperc / 100) * n_mys

tab.loc['Total'] = tab.sum()
tab['Total'] = tab.sum(axis=1)

tabperc.loc['Total'] = tabperc.sum()
tabperc['Total'] = tabperc.sum(axis=1)

tab_scaled.loc['Total'] = tab_scaled.sum()
tab_scaled['Total'] = tab_scaled.sum(axis=1)

tab_scaled = tab_scaled.round(0).astype('int')

# Output
def heatmap_telegram(input, file_name, title):
    fig = heatmap(
        input=input,
        mask=False,
        colourmap='vlag',
        outputfile=file_name + '.png',
        title=title,
        lb=0,
        ub=input.max().max(),
        format='.0f'
    )
    telsendimg(
        conf=tel_config,
        path=file_name + '.png',
        cap=file_name
    )

heatmap_telegram(
    input=tab,
    file_name=path_output + 'tab_target_groups_benefit_incgroup_hies_count',
    title='N of Ind in HIES: By Age and Income Groups'
)
heatmap_telegram(
    input=tabperc,
    file_name=path_output + 'tab_target_groups_benefit_incgroup_hies_perc',
    title='% of Ind in HIES: By Age and Income Groups'
)
heatmap_telegram(
    input=tab_scaled,
    file_name=path_output + 'tab_target_groups_benefit_incgroup_scaled_count',
    title='Ind Scaled to Census with HIES Shares: By Age and Income Groups'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_target_groups_tiered: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
