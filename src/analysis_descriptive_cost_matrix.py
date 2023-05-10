# Not tiered (3 May 2023 version)
# Captive population + immediate cost

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
path_output = './output/'
path_2019 = './data/hies_2019/'
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
        list_incgroups = ['0_b20-', '1_b20+', '2_m20-', '3_m20+', '4_t20']
    elif aggregation == 2:
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
        list_incgroups = ['0_b40', '1_m40', '2_t20']
    return list_incgroups


gen_gross_income_group(data=df, aggregation=1)


# Generate new variables
def gen_hh_size_group(data):
    data['hh_size_group'] = data['hh_size'].copy()
    data.loc[data['hh_size'] >= 8, 'hh_size_group'] = '8+'


def gen_kids(data):
    data['kid'] = df['child'] + df['adolescent']


list_incgroups = gen_gross_income_group(data=df, aggregation=1)
gen_hh_size_group(data=df)
gen_kids(data=df)

# III.0 --- Set parameters for target groups
# Total HH
n_grand_total = 8234644
n_total = len(df)
ngdp_2022 = 1788184000000
child_benefit = 100
working_age_benefit = 50
elderly_benefit = 100
list_amount_multiplier = [1, 1, 0.8, 0.6, 0.4]

# Construct costing matrices (levels and % GDP)
costmat_level = pd.DataFrame(
    columns=['benefit'] + list_incgroups + ['total']
)
costmat_level['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])
costmat_percgdp = pd.DataFrame(
    columns=['benefit'] + list_incgroups + ['total']
)
costmat_percgdp['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])

costmat_partial_level = costmat_level.copy()
costmat_partial_percgdp = costmat_percgdp.copy()

costmat_flat_level = costmat_level.copy()
costmat_flat_percgdp = costmat_percgdp.copy()


# Loop to tabulate shares of households with K kids in Y income group
def tabperc_xy_cost(data, x_col, y_col, list_y, x_max, big_n, mega_n, cost_per_x_per_t, nom_gdp, print_level):
    # Deep copy
    df = data.copy()
    # Blank data frame for percentages
    tabperc = pd.DataFrame()
    # Loop Y group, then by number of X
    for y in list_y:
        for k in range(0, x_max + 1):
            if k < x_max:
                n_group = len(
                    df[
                        (
                                df[x_col] == k
                        ) &
                        (
                                df[y_col] == y
                        )
                        ]
                )
            elif k == x_max:
                n_group = len(
                    df[
                        (
                                df[x_col] >= k
                        ) &
                        (
                                df[y_col] == y
                        )
                        ]
                )
            # Count percentage shares
            perc = 100 * n_group / big_n
            eligible_hh = int((perc / 100) * mega_n)
            eligible_ind = eligible_hh * k
            monthly_cost = eligible_ind * cost_per_x_per_t
            annual_cost = monthly_cost * 12
            annual_cost_gdp = 100 * annual_cost / nom_gdp
            # Compile
            dict_tabperc = {
                y_col: [y],
                x_col: [k],
                'share': [perc],
                'eligible_hh': [eligible_hh],
                'eligible_ind': [eligible_ind],
                'amount_per_ind': [cost_per_x_per_t],
                'monthly_cost': [monthly_cost],
                'annual_cost': [annual_cost],
                'annual_cost_gdp': [annual_cost_gdp]
            }
            tabperc = pd.concat([tabperc, pd.DataFrame(dict_tabperc)])
    # Convert highest k group into range
    tabperc.loc[tabperc[x_col] == x_max, x_col] = str(x_max) + '+'
    # Add one more row for total
    dict_tabperc_total = {
        y_col: ['total'],
        x_col: [x_col],
        'share': [float(tabperc['share'].sum())],
        'eligible_hh': [int(tabperc['eligible_hh'].sum())],
        'eligible_ind': [int(tabperc['eligible_ind'].sum())],
        'amount_per_ind': [int(cost_per_x_per_t)],
        'monthly_cost': [int(tabperc['monthly_cost'].sum())],
        'annual_cost': [int(tabperc['annual_cost'].sum())],
        'annual_cost_gdp': [float(tabperc['annual_cost_gdp'].sum())]
    }
    tabperc = pd.concat([tabperc, pd.DataFrame(dict_tabperc_total)])
    # Flattened version
    cols_flatten = ['share', 'eligible_hh', 'eligible_ind', 'amount_per_ind',
                    'monthly_cost', 'annual_cost', 'annual_cost_gdp']
    tabperc_flattened = tabperc.groupby(y_col)[cols_flatten].sum(numeric_only=True).reset_index(drop=False)
    tabperc_flattened['amount_per_ind'] = cost_per_x_per_t
    # Cleaning table
    cols_round = ['share', 'annual_cost_gdp']
    tabperc[cols_round] = tabperc[cols_round]  # .round(2)
    tabperc_flattened[cols_round] = tabperc_flattened[cols_round]  # .round(2)
    # Print
    if print_level == 0:
        print(
            tabulate(
                tabular_data=tabperc,
                showindex=False,
                headers='keys',
                tablefmt="pretty"
            )
        )
    elif print_level == 1:
        print(
            tabulate(
                tabular_data=tabperc_flattened,
                showindex=False,
                headers='keys',
                tablefmt="pretty"
            )
        )
    elif print_level == -1:
        pass
    # Output
    return tabperc, tabperc_flattened


# III.A --- Flat across all Y group version
# Tabulate kids (0-17) by income group
tabperc_kids_incgroup, tabperc_kids_incgroup_flattened = tabperc_xy_cost(
    data=df,
    x_col='kid',
    y_col='gross_income_group',
    list_y=list_incgroups,
    x_max=7,
    big_n=n_total,
    mega_n=n_grand_total,
    cost_per_x_per_t=child_benefit,
    nom_gdp=ngdp_2022,
    print_level=1
)
tabperc_kids_incgroup['benefit'] = 'kid'
# Tabulate working_age2 (18-64) by income group
tabperc_working_age2_incgroup, tabperc_working_age2_incgroup_flattened = tabperc_xy_cost(
    data=df,
    x_col='working_age2',
    y_col='gross_income_group',
    list_y=list_incgroups,
    x_max=7,
    big_n=n_total,
    mega_n=n_grand_total,
    cost_per_x_per_t=working_age_benefit,
    nom_gdp=ngdp_2022,
    print_level=1
)
tabperc_working_age2_incgroup['benefit'] = 'working_age2'
# Tabulate elderly2 (65+) by income group
tabperc_elderly2_incgroup, tabperc_elderly2_incgroup_flattened = tabperc_xy_cost(
    data=df,
    x_col='elderly2',
    y_col='gross_income_group',
    list_y=list_incgroups,
    x_max=3,
    big_n=n_total,
    mega_n=n_grand_total,
    cost_per_x_per_t=elderly_benefit,
    nom_gdp=ngdp_2022,
    print_level=1
)
tabperc_elderly2_incgroup['benefit'] = 'elderly2'
# Group into single data frame
tabperc_flat_rate = pd.concat([tabperc_kids_incgroup, tabperc_working_age2_incgroup, tabperc_elderly2_incgroup], axis=0)

# III.A --- Tiered by Y group version (flat rate everyone)
for benefit_target, benefit_amount, x_max_choice \
        in \
        zip(['kid', 'working_age2', 'elderly2'], [child_benefit, working_age_benefit, elderly_benefit], [7, 7, 3]):
    for incgroup, amount_multiplier \
            in \
            zip(list_incgroups, [1] * len(list_incgroups)):
        tabperc_flat, tabperc_flat_flattened = tabperc_xy_cost(
            data=df,
            x_col=benefit_target,
            y_col='gross_income_group',
            list_y=[incgroup],
            x_max=x_max_choice,
            big_n=n_total,
            mega_n=n_grand_total,
            cost_per_x_per_t=benefit_amount * amount_multiplier,
            nom_gdp=ngdp_2022,
            print_level=1
        )
        # Input into cost matrix
        costmat_flat_level.loc[costmat_flat_level['benefit'] == benefit_target, incgroup] = \
            tabperc_flat.loc[
                tabperc_flat['gross_income_group'] == incgroup
                ,
                'annual_cost'].sum()
        costmat_flat_percgdp.loc[costmat_flat_percgdp['benefit'] == benefit_target, incgroup] = \
            tabperc_flat.loc[
                tabperc_flat['gross_income_group'] == incgroup
                ,
                'annual_cost_gdp'].sum()

# III.B --- Tiered by Y group version (full tiered)
for benefit_target, benefit_amount, x_max_choice \
        in \
        zip(['kid', 'working_age2', 'elderly2'], [child_benefit, working_age_benefit, elderly_benefit], [7, 7, 3]):
    for incgroup, amount_multiplier \
            in \
            zip(list_incgroups, list_amount_multiplier):
        tabperc_tiered, tabperc_tiered_flattened = tabperc_xy_cost(
            data=df,
            x_col=benefit_target,
            y_col='gross_income_group',
            list_y=[incgroup],
            x_max=x_max_choice,
            big_n=n_total,
            mega_n=n_grand_total,
            cost_per_x_per_t=benefit_amount * amount_multiplier,
            nom_gdp=ngdp_2022,
            print_level=1
        )
        # Input into cost matrix
        costmat_level.loc[costmat_level['benefit'] == benefit_target, incgroup] = \
            tabperc_tiered.loc[
                tabperc_tiered['gross_income_group'] == incgroup
                ,
                'annual_cost'].sum()
        costmat_percgdp.loc[costmat_percgdp['benefit'] == benefit_target, incgroup] = \
            tabperc_tiered.loc[
                tabperc_tiered['gross_income_group'] == incgroup
                ,
                'annual_cost_gdp'].sum()

# III.C --- Tiered by Y group version (partial-tiered)
for benefit_target, benefit_amount, x_max_choice \
        in \
        zip(['kid', 'working_age2', 'elderly2'], [child_benefit, working_age_benefit, elderly_benefit], [7, 7, 3]):
    for incgroup, amount_multiplier \
            in \
            zip(list_incgroups, list_amount_multiplier):
        # Partial tiering
        if (benefit_target == 'kid') | (benefit_target == 'elderly2'):  # which groups to apply flat rate
            amount_multiplier = 1
        # Compute
        tabperc_partial, tabperc_partial_flattened = tabperc_xy_cost(
            data=df,
            x_col=benefit_target,
            y_col='gross_income_group',
            list_y=[incgroup],
            x_max=x_max_choice,
            big_n=n_total,
            mega_n=n_grand_total,
            cost_per_x_per_t=benefit_amount * amount_multiplier,
            nom_gdp=ngdp_2022,
            print_level=1
        )
        # Input into cost matrix
        costmat_partial_level.loc[costmat_partial_level['benefit'] == benefit_target, incgroup] = \
            tabperc_partial.loc[
                tabperc_partial['gross_income_group'] == incgroup
                ,
                'annual_cost'].sum()
        costmat_partial_percgdp.loc[costmat_partial_percgdp['benefit'] == benefit_target, incgroup] = \
            tabperc_partial.loc[
                tabperc_partial['gross_income_group'] == incgroup
                ,
                'annual_cost_gdp'].sum()

# III.D --- Calculate tiered total
# Placeholers
costmat_flat_level['total'] = -1
costmat_flat_percgdp['total'] = -1

costmat_level['total'] = -1
costmat_percgdp['total'] = -1

costmat_partial_level['total'] = -1
costmat_partial_percgdp['total'] = -1
# Setting dtypes
costmat_flat_level[list_incgroups + ['total']] = \
    costmat_flat_level[list_incgroups + ['total']].astype('int64')
costmat_flat_percgdp[list_incgroups + ['total']] = \
    costmat_flat_percgdp[list_incgroups + ['total']].astype('float')

costmat_level[list_incgroups + ['total']] = \
    costmat_level[list_incgroups + ['total']].astype('int64')
costmat_percgdp[list_incgroups + ['total']] = \
    costmat_percgdp[list_incgroups + ['total']].astype('float')

costmat_partial_level[list_incgroups + ['total']] = \
    costmat_partial_level[list_incgroups + ['total']].astype('int64')
costmat_partial_percgdp[list_incgroups + ['total']] = \
    costmat_partial_percgdp[list_incgroups + ['total']].astype('float')

# Compute
costmat_flat_level['total'] = costmat_flat_level[list_incgroups].sum(axis=1)
costmat_flat_percgdp['total'] = costmat_flat_percgdp[list_incgroups].sum(axis=1)

costmat_level['total'] = costmat_level[list_incgroups].sum(axis=1)
costmat_percgdp['total'] = costmat_percgdp[list_incgroups].sum(axis=1)

costmat_partial_level['total'] = costmat_partial_level[list_incgroups].sum(axis=1)
costmat_partial_percgdp['total'] = costmat_partial_percgdp[list_incgroups].sum(axis=1)

# III.D --- Touching up
# Nice column names
costmat_flat_level = costmat_flat_level.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Flat Rate Total',
    }
)
costmat_flat_percgdp = costmat_flat_percgdp.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Flat Rate Total',
    }
)

costmat_level = costmat_level.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Tiered Total',
    }
)
costmat_percgdp = costmat_percgdp.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Tiered Total',
    }
)

costmat_partial_level = costmat_partial_level.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Partial Tiered Total',
    }
)
costmat_partial_percgdp = costmat_partial_percgdp.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'total': 'Partial Tiered Total',
    }
)

# Nice benefit names
costmat_flat_level['benefit'] = costmat_flat_level['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)
costmat_flat_percgdp['benefit'] = costmat_flat_percgdp['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)

costmat_level['benefit'] = costmat_level['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)
costmat_percgdp['benefit'] = costmat_percgdp['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)

costmat_partial_level['benefit'] = costmat_partial_level['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)
costmat_partial_percgdp['benefit'] = costmat_partial_percgdp['benefit'].replace(
    {
        'kid': 'Child (0-17)',
        'working_age2': 'Working Age (18-64)',
        'elderly2': 'Elderly (65+)'
    }
)
# Set index
costmat_flat_level = costmat_flat_level.set_index('benefit')
costmat_flat_percgdp = costmat_flat_percgdp.set_index('benefit')

costmat_level = costmat_level.set_index('benefit')
costmat_percgdp = costmat_percgdp.set_index('benefit')

costmat_partial_level = costmat_partial_level.set_index('benefit')
costmat_partial_percgdp = costmat_partial_percgdp.set_index('benefit')

# Add column total
costmat_flat_level.loc['All Benefits'] = costmat_flat_level.sum(axis=0)
costmat_flat_percgdp.loc['All Benefits'] = costmat_flat_percgdp.sum(axis=0)

costmat_level.loc['All Benefits'] = costmat_level.sum(axis=0)
costmat_percgdp.loc['All Benefits'] = costmat_percgdp.sum(axis=0)

costmat_partial_level.loc['All Benefits'] = costmat_partial_level.sum(axis=0)
costmat_partial_percgdp.loc['All Benefits'] = costmat_partial_percgdp.sum(axis=0)
# Convert to RM bil
costmat_flat_level = costmat_flat_level / 1000000000
costmat_level = costmat_level / 1000000000
costmat_partial_level = costmat_partial_level / 1000000000

# III.E --- Show only total cost
costmat_allcombos_level = pd.concat(
    [
        costmat_level[['Tiered Total']],
        costmat_partial_level[['Partial Tiered Total']],
        costmat_flat_level[['Flat Rate Total']]
    ],
    axis=1
)
costmat_allcombos_percgdp = pd.concat(
    [
        costmat_percgdp[['Tiered Total']],
        costmat_partial_percgdp[['Partial Tiered Total']],
        costmat_flat_percgdp[['Flat Rate Total']]
    ],
    axis=1
)

# III.F --- Output locally (before rounding)
costmat_flat_level.to_parquet(path_output + 'cost_matrix_flat_level.parquet')
costmat_flat_percgdp.to_parquet(path_output + 'cost_matrix_flat_percgdp.parquet')
costmat_flat_level.to_csv(path_output + 'cost_matrix_flat_level.csv')
costmat_flat_percgdp.to_csv(path_output + 'cost_matrix_flat_percgdp.csv')

costmat_level.to_parquet(path_output + 'cost_matrix_level.parquet')
costmat_percgdp.to_parquet(path_output + 'cost_matrix_percgdp.parquet')
costmat_level.to_csv(path_output + 'cost_matrix_level.csv')
costmat_percgdp.to_csv(path_output + 'cost_matrix_percgdp.csv')

costmat_partial_level.to_parquet(path_output + 'cost_matrix_partial_level.parquet')
costmat_partial_percgdp.to_parquet(path_output + 'cost_matrix_partial_percgdp.parquet')
costmat_partial_level.to_csv(path_output + 'cost_matrix_partial_level.csv')
costmat_partial_percgdp.to_csv(path_output + 'cost_matrix_partial_percgdp.csv')

costmat_allcombos_level.to_parquet(path_output + 'cost_matrix_allcombos_level.parquet')
costmat_allcombos_percgdp.to_parquet(path_output + 'cost_matrix_allcombos_percgdp.parquet')

# III.G --- Visualisation
# Round up values
costmat_flat_level = costmat_flat_level.round(2)
costmat_flat_percgdp = costmat_flat_percgdp.round(2)

costmat_level = costmat_level.round(2)
costmat_percgdp = costmat_percgdp.round(2)

costmat_partial_level = costmat_partial_level.round(2)
costmat_partial_percgdp = costmat_partial_percgdp.round(2)

costmat_allcombos_level = costmat_allcombos_level.round(2)
costmat_allcombos_percgdp = costmat_allcombos_percgdp.round(2)

# Convert into heatmap
fig_costmat_flat_level = heatmap(
    input=costmat_flat_level,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_flat_level.png',
    title='Annual Cost (RM bil): Flat Rate',
    lb=0,
    ub=costmat_flat_level.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_flat_level.png',
    cap='cost_matrix_flat_level'
)

fig_costmat_flat_percgdp = heatmap(
    input=costmat_flat_percgdp,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_flat_percgdp.png',
    title='Annual Cost (% of 2022 GDP): Flat Rate',
    lb=0,
    ub=costmat_flat_percgdp.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_flat_percgdp.png',
    cap='cost_matrix_flat_percgdp'
)

fig_costmat_level = heatmap(
    input=costmat_level,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_level.png',
    title='Annual Cost (RM bil): Full Tiering',
    lb=0,
    ub=costmat_level.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_level.png',
    cap='cost_matrix_level'
)

fig_costmat_percgdp = heatmap(
    input=costmat_percgdp,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_percgdp.png',
    title='Annual Cost (% of 2022 GDP): Full Tiering',
    lb=0,
    ub=costmat_percgdp.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_percgdp.png',
    cap='cost_matrix_percgdp'
)

fig_costmat_partial_level = heatmap(
    input=costmat_partial_level,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_partial_level.png',
    title='Annual Cost (RM bil): Partial Tiering',
    lb=0,
    ub=costmat_partial_level.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_partial_level.png',
    cap='cost_matrix_partial_level'
)

fig_costmat_partial_percgdp = heatmap(
    input=costmat_partial_percgdp,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_partial_percgdp.png',
    title='Annual Cost (% of 2022 GDP): Partial Tiering',
    lb=0,
    ub=costmat_partial_percgdp.max().max(),
    format='.2f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_partial_percgdp.png',
    cap='cost_matrix_partial_percgdp'
)

fig_costmat_allcombos_level = heatmap(
    input=costmat_allcombos_level,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_allcombos_level.png',
    title='Annual Cost (RM bil): All Combinations',
    lb=0,
    ub=costmat_allcombos_level.max().max(),
    format='.1f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_allcombos_level.png',
    cap='cost_matrix_allcombos_level'
)

fig_costmat_allcombos_percgdp = heatmap(
    input=costmat_allcombos_percgdp,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_allcombos_percgdp.png',
    title='Annual Cost (% of 2022 GDP): All Combinations',
    lb=0,
    ub=costmat_allcombos_percgdp.max().max(),
    format='.1f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_allcombos_percgdp.png',
    cap='cost_matrix_allcombos_percgdp'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_cost_matrix: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
