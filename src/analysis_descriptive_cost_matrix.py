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
    columns=['benefit'] + list_incgroups + ['tiered_total', 'flat_rate_total']
)
costmat_level['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])

costmat_percgdp = pd.DataFrame(
    columns=['benefit'] + list_incgroups + ['tiered_total', 'flat_rate_total']
)
costmat_percgdp['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])


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
# Input into blank cost matrices
for x in ['kid', 'working_age2', 'elderly2']:
    costmat_level.loc[costmat_level['benefit'] == x, 'flat_rate_total'] = \
        tabperc_flat_rate.loc[(
                                      (tabperc_flat_rate['benefit'] == x) &
                                      (tabperc_flat_rate['gross_income_group'] == 'total')
                              ),
                              'annual_cost'].sum()
    costmat_percgdp.loc[costmat_percgdp['benefit'] == x, 'flat_rate_total'] = \
        tabperc_flat_rate.loc[(
                                      (tabperc_flat_rate['benefit'] == x) &
                                      (tabperc_flat_rate['gross_income_group'] == 'total')
                              ),
                              'annual_cost_gdp'].sum()

# III.B --- Tiered by Y group version
for benefit_target, benefit_amount, x_max_choice \
        in \
        zip(['kid', 'working_age2', 'elderly2'], [child_benefit, working_age_benefit, elderly_benefit], [7, 7, 3]):
    for incgroup, amount_multiplier \
            in \
            zip(list_incgroups, list_amount_multiplier):
        # Compute
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

# III.C --- Calculate tiered total
# Placeholers
costmat_percgdp['tiered_total'] = -1
costmat_level['tiered_total'] = -1
# Setting dtypes
costmat_level[list_incgroups + ['tiered_total', 'flat_rate_total']] = \
    costmat_level[list_incgroups + ['tiered_total', 'flat_rate_total']].astype('int64')
costmat_percgdp[list_incgroups + ['tiered_total', 'flat_rate_total']] = \
    costmat_percgdp[list_incgroups + ['tiered_total', 'flat_rate_total']].astype('float')
# Compute
costmat_level['tiered_total'] = costmat_level[list_incgroups].sum(axis=1)
costmat_percgdp['tiered_total'] = costmat_percgdp[list_incgroups].sum(axis=1)

# III.X --- Touching up
# Nice column names
costmat_level = costmat_level.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'tiered_total': 'Tiered Total',
        'flat_rate_total': 'Flat Rate Total'
    }
)
costmat_percgdp = costmat_percgdp.rename(
    columns={
        '0_b20-': 'B20-',
        '1_b20+': 'B20+',
        '2_m20-': 'M20-',
        '3_m20+': 'M20+',
        '4_t20': 'T20',
        'tiered_total': 'Tiered Total',
        'flat_rate_total': 'Flat Rate Total'
    }
)
# Nice benefit names
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
# Set index
costmat_level = costmat_level.set_index('benefit')
costmat_percgdp = costmat_percgdp.set_index('benefit')
# Add column total
costmat_level.loc['All Benefits'] = costmat_level.sum(axis=0)
costmat_percgdp.loc['All Benefits'] = costmat_percgdp.sum(axis=0)
# Convert to RM bil
costmat_level = costmat_level / 1000000000
# Round up values
costmat_level = costmat_level.round(2)
costmat_percgdp = costmat_percgdp.round(2)
# Print for inspection
print(
    tabulate(
        tabular_data=costmat_level,
        showindex=True,
        headers='keys',
        tablefmt="pretty"
    )
)
print(
    tabulate(
        tabular_data=costmat_percgdp,
        showindex=True,
        headers='keys',
        tablefmt="pretty"
    )
)

# III.X --- Convert into heatmap
fig_costmat_level = heatmap(
    input=costmat_level,
    mask=False,
    colourmap='coolwarm',
    outputfile=path_output + 'cost_matrix_level.png',
    title='Annual Cost Matrix (RM bil)',
    lb=0,
    ub=costmat_level.max().max(),
    format='.1f'
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
    title='Annual Cost Matrix (% of 2022 GDP)',
    lb=0,
    ub=costmat_percgdp.max().max(),
    format='.1f'
)
telsendimg(
    conf=tel_config,
    path=path_output + 'cost_matrix_percgdp.png',
    cap='cost_matrix_percgdp'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_cost_matrix: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
