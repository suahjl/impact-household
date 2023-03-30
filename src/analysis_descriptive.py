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
use_spending_income_ratio = ast.literal_eval(os.getenv('USE_SPENDING_INCOME_RATIO'))

# I --- Load data
df = pd.read_parquet(
    path_data + 'hies_consol_ind_full_hhbasis.parquet')  # CHECK: include / exclude outliers and on hhbasis

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
        'adolescent_group',
        'child_group',
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
                ['cons_01', 'cons_04', 'cons_06', 'cons_07',
                 'cons_10']  # F&B, util, healthcare, transport & fuels, education
# ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']
# Define list of quantiles
list_quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                  0.95]
list_quantiles_nice = ['5p', '10p', '15p', '20p', '25p', '30p', '35p', '40p', '45p', '50p', '55p', '60p', '65p', '70p',
                       '75p', '80p', '85p', '90p', '95p']
# Colour scheme for time
list_colours_time = ['black', 'darkblue', 'crimson']

# Convert consumption into % of income
list_all_cons = ['cons_01_13', 'cons_01_12'] + \
                ['cons_0' + str(i) for i in range(1, 10)] + \
                ['cons_1' + str(i) for i in range(0, 4)]
if use_spending_income_ratio:
    for cons in list_all_cons:
        df[cons] = 100 * df[cons] / df['gross_income']

# III.0 --- Define function


def capybara(data, y_cols, x_cols, t_col, quantiles, quantiles_nice):
    # Prelims
    d_full = data.copy()
    list_t = list(d_full[t_col].unique())
    list_t.sort()
    # Tabulate y by x and t (y must be continuous, and x must be categorical)
    tab_consol = pd.DataFrame(columns=quantiles_nice)  # empty master dataframe
    for y in tqdm(y_cols):
        tab_yqtx = pd.DataFrame([[y] * len(quantiles)], columns=quantiles_nice)
        tab_consol = pd.concat([tab_consol, tab_yqtx], axis=0)
        for x in x_cols:
            tab_qtx = pd.DataFrame([[x] * len(quantiles)], columns=quantiles_nice)
            tab_consol = pd.concat([tab_consol, tab_qtx], axis=0)
            for t in list_t:
                d = d_full[d_full[t_col] == t]
                tab_qt = pd.DataFrame([[t] * len(quantiles)], columns=quantiles_nice)
                tab_consol = pd.concat([tab_consol, tab_qt], axis=0)
                round_q = 1
                for quantile in quantiles:
                    tab = d.groupby(x)[y].quantile(q=quantile)
                    if round_q == 1:
                        tab_q = tab.copy()
                    elif round_q > 1:
                        tab_q = pd.concat([tab_q, tab], axis=1)  # left-right
                    round_q += 1
                tab_q.columns = quantiles_nice
                tab_consol = pd.concat([tab_consol, tab_q], axis=0)
    # Output
    return tab_consol


def capybara_mini(data, y_col, x_col, quantiles, quantiles_nice):
    # Prelims
    d = data.copy()
    # Tabulate y (y must be continuous, and x must be categorical)
    round_q = 1
    for quantile in quantiles:
        tab = d.groupby(x_col)[y_col].quantile(q=quantile)
        if round_q == 1:
            tab_q = tab.copy()
        elif round_q > 1:
            tab_q = pd.concat([tab_q, tab], axis=1)  # left-right
        round_q += 1
    tab_q.columns = quantiles_nice
    # Harmonise format
    tab_q = tab_q.astype('float')
    # Output
    return tab_q


def capybara_chonky(data, y_col, x_col):
    # Prelims
    d = data.copy()
    # Cross tabulate (both y and x must be categorical)
    tab = pd.crosstab(d[y_col], d[x_col], normalize=False)  # rows = outcome, columns = categories
    tabperc = 100 * pd.crosstab(d[y_col], d[x_col], normalize='columns')  # % distribution within y variables
    # Harmonise format
    tab = tab.astype('int')
    tabperc = tabperc.astype('float')
    # Output
    return tab, tabperc


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


def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 10, 'hh_size_group'] = '10+'


# III.0 --- Generate income groups
# Define income group buckets
gen_gross_income_group(data=df, aggregation=1)
# Define hh size group buckets
gen_hh_size_group(data=df)

# III.A --- Stratify selected categorical outcomes by income group and by time (only heat tables)

for y in tqdm(list_cat_outcomes):
    # Heat tables
    list_heattable_names = []  # reset every categorical outcome (too many otherwise)
    for t in df['year'].unique():
        tab_obs_incgroup, tabperc_obs_incgroup = capybara_chonky(
            data=df[df['year'] == t],
            y_col=y,
            x_col='gross_income_group'
        )
        heattable_name = 'output/tab_' + y + '_' + 'gross_income_group' + '_' + str(t)
        heattable_perc_name = 'output/tabperc_' + y + '_' + 'gross_income_group' + '_' + str(t)
        list_heattable_names = list_heattable_names + [heattable_name, heattable_perc_name]
        fig_tab_obs_incgroup = heatmap(
            input=tab_obs_incgroup,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_name + '.png',
            title='Cross tab (N) of ' + y + ' by ' + 'gross_income_group' + ' for year ' + str(t),
            lb=tab_obs_incgroup.min().max(),
            ub=tab_obs_incgroup.max().max(),
            format='.0f'
        )
        fig_tabperc_obs_incgroup = heatmap(
            input=tabperc_obs_incgroup,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_perc_name + '.png',
            title='Cross tab (%) of ' + y + ' by ' + 'gross_income_group' + ' for year ' + str(t),
            lb=tabperc_obs_incgroup.min().max(),
            ub=tabperc_obs_incgroup.max().max(),
            format='.1f'
        )
    pil_img2pdf(
        list_images=list_heattable_names,
        extension='png', pdf_name='output/tab_tabperc_' + y + '_incgroup_years'
    )
    # telsendfiles(
    #     conf=tel_config,
    #     path='output/tab_tabperc_' + y + '_incgroup_years.pdf',
    #     cap='tab_tabperc_' + y + '_incgroup_years'
    # )

# III.B --- Stratify quantiles of consumption type and income type, by income groups and by time (heat tables & box)
# Generate heat tables (y by income groups)
list_boxplot_names = []
list_latest_median_names = []
list_heattable_names = []
for y in tqdm(list_outcomes):
    # Box plots
    boxplot_marginsinccons_incgroup = boxplot_time(
        data=df,
        y_col=y,
        x_col='gross_income_group',
        t_col='year',
        colours=list_colours_time,
        main_title='Boxplot of ' + y + ' by ' + 'gross_income_group' + ' over time'
    )
    boxplot_name = 'output/boxplot_' + y + '_incgroup_years'
    list_boxplot_names = list_boxplot_names + [boxplot_name]
    boxplot_marginsinccons_incgroup.write_image('output/boxplot_' + y + '_incgroup_years.png')
    # Latest medians
    latest_median_marginsinccons_incgroup = df[df['year'] == 2019].groupby('gross_income_group')[y].median().reset_index()
    latest_median_name = 'output/latest_median_' + y + '_incgroup_years'
    list_latest_median_names = list_latest_median_names + [latest_median_name]
    dfi.export(latest_median_marginsinccons_incgroup, latest_median_name + '.png')
    # Heat maps
    for t in df['year'].unique():
        tab_marginsinccons_incgroup = capybara_mini(
            data=df[df['year'] == t],
            y_col=y,
            x_col='gross_income_group',
            quantiles=list_quantiles,
            quantiles_nice=list_quantiles_nice
        )
        tab_marginsinccons_incgroup = tab_marginsinccons_incgroup.transpose()  # more space horizontally for values
        heattable_name = 'output/tab_' + y + '_' + 'gross_income_group' + '_' + str(t)
        list_heattable_names = list_heattable_names + [heattable_name]
        fig_marginsinccons_incgroup = heatmap(
            input=tab_marginsinccons_incgroup,
            mask=False,
            colourmap='vlag',
            outputfile=heattable_name + '.png',
            title='Quantiles of ' + y + ' by ' + 'gross_income_group' + ' for year ' + str(t),
            lb=tab_marginsinccons_incgroup.min().max(),
            ub=tab_marginsinccons_incgroup.max().max(),
            format='.0f'
        )
pil_img2pdf(
    list_images=list_heattable_names,
    extension='png', pdf_name='output/tab_marginsinccons_incgroup_years'
)
telsendfiles(
    conf=tel_config,
    path='output/tab_marginsinccons_incgroup_years.pdf',
    cap='tab_marginsinccons_incgroup_years'
)
# Continue boxplots
pil_img2pdf(
    list_images=list_boxplot_names,
    extension='png',
    pdf_name='output/boxplot_marginsinccons_incgroup_years'
)
telsendfiles(
    conf=tel_config,
    path='output/boxplot_marginsinccons_incgroup_years.pdf',
    cap='boxplot_marginsinccons_incgroup_years'
)
# Continue latest medians
pil_img2pdf(
    list_images=list_latest_median_names,
    extension='png',
    pdf_name='output/latest_median_marginsinccons_incgroup_years'
)
telsendfiles(
    conf=tel_config,
    path='output/latest_median_marginsinccons_incgroup_years.pdf',
    cap='latest_median_marginsinccons_incgroup_years'
)

# III.C --- Stratify quantiles of consumption type and income type, by observables and by time
# Generate heat tables (y by categorical observables)
for x in tqdm(list_groups):
    list_boxplot_names = []  # reset every x loop
    list_heattable_names = []  # reset every x loop
    for y in list_outcomes:
        # Box plots
        boxplot_marginsinccons_obs = boxplot_time(
            data=df,
            y_col=y,
            x_col=x,
            t_col='year',
            colours=list_colours_time,
            main_title='Boxplot of ' + y + ' by ' + x + ' over time'
        )
        boxplot_name = 'output/boxplot_' + y + '_' + x + '_years'
        list_boxplot_names = list_boxplot_names + [boxplot_name]
        boxplot_marginsinccons_obs.write_image('output/boxplot_' + y + '_' + x + '_years.png')
        # Latest medians
        latest_median_marginsinccons_obs = df[df['year'] == 2019].groupby(x)[y].median().reset_index()
        latest_median_name = 'output/latest_median_' + y + '_' + x + '_years'
        list_latest_median_names = list_latest_median_names + [latest_median_name]
        dfi.export(latest_median_marginsinccons_obs, latest_median_name + '.png')
        # Heatmaps
        for t in df['year'].unique():
            tab_marginsinccons_obs = capybara_mini(
                data=df[df['year'] == t],
                y_col=y,
                x_col=x,
                quantiles=list_quantiles,
                quantiles_nice=list_quantiles_nice
            )
            tab_marginsinccons_obs = tab_marginsinccons_obs.transpose()  # more space horizontally for values
            heattable_name = 'output/tab_' + y + '_' + x + '_' + str(t)
            list_heattable_names = list_heattable_names + [heattable_name]
            fig_marginsinccons_incgroup = heatmap(
                input=tab_marginsinccons_obs,
                mask=False,
                colourmap='vlag',
                outputfile=heattable_name + '.png',
                title='Quantiles of ' + y + ' by ' + x + ' for year ' + str(t),
                lb=tab_marginsinccons_obs.min().max(),
                ub=tab_marginsinccons_obs.max().max(),
                format='.0f'
            )
    pil_img2pdf(
        list_images=list_heattable_names,
        extension='png', pdf_name='output/tab_marginsinccons_' + x + '_years'
    )
    # telsendfiles(
    #     conf=tel_config,
    #     path='output/tab_marginsinccons_' + x + '_years.pdf',
    #     cap='tab_marginsinccons_' + x + '_years'
    # )
    # Continue boxplots
    pil_img2pdf(
        list_images=list_boxplot_names,
        extension='png', pdf_name='output/boxplot_marginsinccons_' + x + '_years'
    )
    # telsendfiles(
    #     conf=tel_config,
    #     path='output/boxplot_marginsinccons_' + x + '_years.pdf',
    #     cap='boxplot_marginsinccons_' + x + '_years.pdf'
    # )
    # Continue latest medians
    pil_img2pdf(
        list_images=list_latest_median_names,
        extension='png',
        pdf_name='output/latest_median_marginsinccons_' + x + '_years'
    )
    # telsendfiles(
    #     conf=tel_config,
    #     path='output/latest_median_marginsinccons_' + x + '_years.pdf',
    #     cap='latest_median_marginsinccons_' + x + '_years'
    # )

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
