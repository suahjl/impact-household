# Redone for computational efficiency

import pandas as pd
import numpy as np
from helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols, heatmap, pil_img2pdf, boxplot_time
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
path_data = 'data/hies_consol/'
path_output = 'output/'
path_2022 = 'data/hies_2022/'
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
    path_2022 + 'hies_2022_consol' + input_suffix + '.parquet')  # CHECK: include / exclude outliers and on hhbasis

# II --- Pre-analysis prep
# Malaysian only


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

# III --- Set parameters for target groups
# Total HH
n_grand_total = 8234644
n_total = len(df)
ngdp_2022 = 1788184000000
child_benefit = 100
working_age_benefit = 50
elderly_benefit = 100
tiered_amount_multiplier = [1, 1, 0.8, 0.6, 0.4]


# Function: Loop to tabulate shares of households with K characteristics (X) in Y income group
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


# Construct costing matrices (levels and % GDP)
costmat_base_level = pd.DataFrame(
    columns=['benefit'] + list_incgroups + ['total']
)
costmat_base_level['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])

costmat_base_percgdp = pd.DataFrame(
    columns=['benefit'] + list_incgroups + ['total']
)
costmat_base_percgdp['benefit'] = pd.Series(['kid', 'working_age2', 'elderly2'])


# Function: Compute scenario-specific cost
def compute_scenario_cost(
        multiplier_set,
        tier_wa,
        threshold_exclude_wa,
        tier_kids,
        threshold_exclude_kids,
        tier_elderly,
        threshold_exclude_elderly
):
    costmat_level = costmat_base_level.copy()
    costmat_percgdp = costmat_base_percgdp.copy()
    for benefit_target, benefit_amount, x_max_choice \
            in \
            zip(
                ['kid', 'working_age2', 'elderly2'],
                [child_benefit, working_age_benefit, elderly_benefit],
                [7, 7, 3]
            ):
        count_incgroup = 1
        for incgroup, amount_multiplier \
                in \
                zip(
                    list_incgroups,
                    multiplier_set
                ):
            # Check if tiered or flat for current round
            # WA
            if (benefit_target == 'working_age2') & (not tier_wa):
                amount_multiplier = 1  # flat rate
            # Kids
            if (benefit_target == 'kid') & (not tier_kids):
                amount_multiplier = 1  # flat rate
            # Elderly
            if (benefit_target == 'elderly2') & (not tier_elderly):
                amount_multiplier = 1  # flat rate

            # Check if current income group is restricted from specific benefits
            # WA
            if not threshold_exclude_wa:
                pass  # no exclusion, follow input multiplier
            elif threshold_exclude_wa:
                if (count_incgroup >= threshold_exclude_wa) & (benefit_target == 'working_age2'):
                    amount_multiplier = 0
            # Kids
            if not threshold_exclude_kids:
                pass  # no exclusion, follow input multiplier
            elif threshold_exclude_kids:
                if (count_incgroup >= threshold_exclude_kids) & (benefit_target == 'kid'):
                    amount_multiplier = 0
            # Elderly
            if not threshold_exclude_elderly:
                pass  # no exclusion, follow input multiplier
            elif threshold_exclude_elderly:
                if (count_incgroup >= threshold_exclude_elderly) & (benefit_target == 'elderly2'):
                    amount_multiplier = 0

            # Compute cost and cost breakdown
            tabperc, tabperc_flattened = tabperc_xy_cost(
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
                tabperc.loc[
                    tabperc['gross_income_group'] == incgroup
                    ,
                    'annual_cost'].sum()
            costmat_percgdp.loc[costmat_percgdp['benefit'] == benefit_target, incgroup] = \
                tabperc.loc[
                    tabperc['gross_income_group'] == incgroup
                    ,
                    'annual_cost_gdp'].sum()

            # Increase count for income group
            count_incgroup += 1

    # Calculate row total
    costmat_level['total'] = -1
    costmat_percgdp['total'] = -1

    costmat_level[list_incgroups + ['total']] = \
        costmat_level[list_incgroups + ['total']].astype('int64')
    costmat_percgdp[list_incgroups + ['total']] = \
        costmat_percgdp[list_incgroups + ['total']].astype('float')

    costmat_level['total'] = costmat_level[list_incgroups].sum(axis=1)
    costmat_percgdp['total'] = costmat_percgdp[list_incgroups].sum(axis=1)

    # Nice column names
    costmat_level = costmat_level.rename(
        columns={
            '0_b20-': 'B20-',
            '1_b20+': 'B20+',
            '2_m20-': 'M20-',
            '3_m20+': 'M20+',
            '4_t20': 'T20',
            'total': 'Total',
        }
    )
    costmat_percgdp = costmat_percgdp.rename(
        columns={
            '0_b20-': 'B20-',
            '1_b20+': 'B20+',
            '2_m20-': 'M20-',
            '3_m20+': 'M20+',
            '4_t20': 'T20',
            'total': 'Total',
        }
    )
    # Nice benefit row names
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

    # Output
    return costmat_level, costmat_percgdp


# IV --- Compute, export breakdowns, and aggregate all combos
# A --- Set up group restrictions
dict_scenarios_wa_restrict = {
    # 'B40': 3,
    'B60': 4,
    'B80': 5,
    'All': False
}
# B --- Universal and flat for kids and elderly, tier and restrict WA
round_scenario = 1
list_file_names = []
for scenario_name, choice_threshold_exclude_wa in tqdm(dict_scenarios_wa_restrict.items()):
    # Compute
    costmat_level, costmat_percgdp = compute_scenario_cost(
        multiplier_set=tiered_amount_multiplier,
        tier_wa=True,
        tier_kids=False,
        tier_elderly=False,
        threshold_exclude_wa=choice_threshold_exclude_wa,
        threshold_exclude_kids=False,
        threshold_exclude_elderly=False
    )
    # File names of scenario-specific breakdown
    file_name_level = path_output + \
                      'cost_matrix_' + \
                      'kids_elderly_flat_universal_' + \
                      'wa_tiered_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_level'
    file_name_percgdp = path_output + \
                        'cost_matrix_' + \
                        'kids_elderly_flat_universal_' + \
                        'wa_tiered_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_percgdp'
    # Export parquet and csv of scenario-specific breakdown
    costmat_level.to_parquet(file_name_level + '.parquet')
    costmat_level.to_csv(file_name_level + '.csv', index=False)
    costmat_percgdp.to_parquet(file_name_percgdp + '.parquet')
    costmat_percgdp.to_csv(file_name_percgdp + '.csv', index=False)
    # Heatmap of scenario-specific breakdown
    fig_costmat_level = heatmap(
        input=costmat_level,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_level + '.png',
        title='WA: Tiered and ' + scenario_name + '; Kids & Elderly: Flat and Universal',
        lb=0,
        ub=costmat_level.max().max(),
        format='.1f'
    )
    fig_costmat_percgdp = heatmap(
        input=costmat_percgdp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_percgdp + '.png',
        title='WA: Tiered and ' + scenario_name + '; Kids & Elderly: Flat and Universal; (% GDP)',
        lb=0,
        ub=costmat_percgdp.max().max(),
        format='.1f'
    )
    list_file_names = list_file_names + [file_name_level, file_name_percgdp]
    # Consolidate
    if round_scenario == 1:
        costmat_level_consol = pd.DataFrame(costmat_level['Total']).copy()
        costmat_percgdp_consol = pd.DataFrame(costmat_percgdp['Total']).copy()
    if round_scenario > 1:
        costmat_level_consol = pd.concat(
            [costmat_level_consol, costmat_level['Total']],
            axis=1)  # left-right concat
        costmat_percgdp_consol = pd.concat(
            [costmat_percgdp_consol, costmat_percgdp['Total']],
            axis=1)  # left-right concat
    # Distinguish new columns
    costmat_level_consol = costmat_level_consol.rename(columns={'Total': scenario_name})
    costmat_percgdp_consol = costmat_percgdp_consol.rename(columns={'Total': scenario_name})
    round_scenario += 1
# Heatmap for consolidated version
file_name_level = path_output + \
                  'cost_matrix_' + \
                  'kids_elderly_flat_universal_' + \
                  'wa_tiered_restrict' + \
                  '_level'
file_name_percgdp = path_output + \
                    'cost_matrix_' + \
                    'kids_elderly_flat_universal_' + \
                    'wa_tiered_restrict' + \
                    '_percgdp'
fig_costmat_level_consol = heatmap(
    input=costmat_level_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_level + '.png',
    title='WA: Tiered and Restricted' + '; Kids & Elderly: Flat and Universal',
    lb=0,
    ub=costmat_level_consol.max().max(),
    format='.1f'
)
fig_costmat_percgdp_consol = heatmap(
    input=costmat_percgdp_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_percgdp + '.png',
    title='WA: Tiered and Restricted' + '; Kids & Elderly: Flat and Universal; (% GDP)',
    lb=0,
    ub=costmat_percgdp_consol.max().max(),
    format='.1f'
)
# Generate parquet and csv files for consolidated disbursement table
costmat_level_consol.to_parquet(file_name_level + '.parquet')
costmat_level_consol.to_csv(file_name_level + '.csv', index=False)
costmat_percgdp_consol.to_parquet(file_name_percgdp + '.parquet')
costmat_percgdp_consol.to_csv(file_name_percgdp + '.csv', index=False)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate consolidated PDF
pil_img2pdf(list_images=list_file_names,
            extension='png',
            pdf_name=path_output + 'cost_matrix_' +
                     'kids_elderly_flat_universal_' +
                     'wa_tiered_restrict' +
                     '_consol')

# C --- Universal and flat for kids and elderly, flat but restrict WA
round_scenario = 1
list_file_names = []
for scenario_name, choice_threshold_exclude_wa in tqdm(dict_scenarios_wa_restrict.items()):
    # Compute
    costmat_level, costmat_percgdp = compute_scenario_cost(
        multiplier_set=tiered_amount_multiplier,
        tier_wa=False,  # flat rate
        tier_kids=False,
        tier_elderly=False,
        threshold_exclude_wa=choice_threshold_exclude_wa,
        threshold_exclude_kids=False,
        threshold_exclude_elderly=False
    )
    # File names of scenario-specific breakdown
    file_name_level = path_output + \
                      'cost_matrix_' + \
                      'kids_elderly_flat_universal_' + \
                      'wa_flat_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_level'
    file_name_percgdp = path_output + \
                        'cost_matrix_' + \
                        'kids_elderly_flat_universal_' + \
                        'wa_flat_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_percgdp'
    # Export parquet and csv of scenario-specific breakdown
    costmat_level.to_parquet(file_name_level + '.parquet')
    costmat_level.to_csv(file_name_level + '.csv', index=False)
    costmat_percgdp.to_parquet(file_name_percgdp + '.parquet')
    costmat_percgdp.to_csv(file_name_percgdp + '.csv', index=False)
    # Heatmap of scenario-specific breakdown
    fig_costmat_level = heatmap(
        input=costmat_level,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_level + '.png',
        title='WA: Flat and ' + scenario_name + '; Kids & Elderly: Flat and Universal',
        lb=0,
        ub=costmat_level.max().max(),
        format='.1f'
    )
    fig_costmat_percgdp = heatmap(
        input=costmat_percgdp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_percgdp + '.png',
        title='WA: Flat and ' + scenario_name + '; Kids & Elderly: Flat and Universal; (% GDP)',
        lb=0,
        ub=costmat_percgdp.max().max(),
        format='.1f'
    )
    list_file_names = list_file_names + [file_name_level, file_name_percgdp]
    # Consolidate
    if round_scenario == 1:
        costmat_level_consol = pd.DataFrame(costmat_level['Total']).copy()
        costmat_percgdp_consol = pd.DataFrame(costmat_percgdp['Total']).copy()
    if round_scenario > 1:
        costmat_level_consol = pd.concat(
            [costmat_level_consol, costmat_level['Total']],
            axis=1)  # left-right concat
        costmat_percgdp_consol = pd.concat(
            [costmat_percgdp_consol, costmat_percgdp['Total']],
            axis=1)  # left-right concat
    # Distinguish new columns
    costmat_level_consol = costmat_level_consol.rename(columns={'Total': scenario_name})
    costmat_percgdp_consol = costmat_percgdp_consol.rename(columns={'Total': scenario_name})
    round_scenario += 1
# Heatmap for consolidated version
file_name_level = path_output + \
                  'cost_matrix_' + \
                  'kids_elderly_flat_universal_' + \
                  'wa_flat_restrict' + \
                  '_level'
file_name_percgdp = path_output + \
                    'cost_matrix_' + \
                    'kids_elderly_flat_universal_' + \
                    'wa_flat_restrict' + \
                    '_percgdp'
fig_costmat_level_consol = heatmap(
    input=costmat_level_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_level + '.png',
    title='WA: Flat and Restricted' + '; Kids & Elderly: Flat and Universal',
    lb=0,
    ub=costmat_level_consol.max().max(),
    format='.1f'
)
fig_costmat_percgdp_consol = heatmap(
    input=costmat_percgdp_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_percgdp + '.png',
    title='WA: Flat and Restricted' + '; Kids & Elderly: Flat and Universal; (% GDP)',
    lb=0,
    ub=costmat_percgdp_consol.max().max(),
    format='.1f'
)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate parquet and csv files for consolidated disbursement table
costmat_level_consol.to_parquet(file_name_level + '.parquet')
costmat_level_consol.to_csv(file_name_level + '.csv', index=False)
costmat_percgdp_consol.to_parquet(file_name_percgdp + '.parquet')
costmat_percgdp_consol.to_csv(file_name_percgdp + '.csv', index=False)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate consolidated PDF
pil_img2pdf(list_images=list_file_names,
            extension='png',
            pdf_name=path_output + 'cost_matrix_' +
                     'kids_elderly_flat_universal_' +
                     'wa_flat_restrict' +
                     '_consol')

# D --- Restricted and flat for everything
round_scenario = 1
list_file_names = []
for scenario_name, choice_threshold_exclude_wa in tqdm(dict_scenarios_wa_restrict.items()):
    # Compute
    costmat_level, costmat_percgdp = compute_scenario_cost(
        multiplier_set=tiered_amount_multiplier,
        tier_wa=False,  # flat rate
        tier_kids=False,
        tier_elderly=False,
        threshold_exclude_wa=choice_threshold_exclude_wa,  # restrict
        threshold_exclude_kids=choice_threshold_exclude_wa,
        threshold_exclude_elderly=choice_threshold_exclude_wa
    )
    # File names of scenario-specific breakdown
    file_name_level = path_output + \
                      'cost_matrix_' + \
                      'kids_elderly_flat_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_' + \
                      'wa_flat_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_level'
    file_name_percgdp = path_output + \
                        'cost_matrix_' + \
                        'kids_elderly_flat_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_' + \
                        'wa_flat_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_percgdp'
    # Export parquet and csv of scenario-specific breakdown
    costmat_level.to_parquet(file_name_level + '.parquet')
    costmat_level.to_csv(file_name_level + '.csv', index=False)
    costmat_percgdp.to_parquet(file_name_percgdp + '.parquet')
    costmat_percgdp.to_csv(file_name_percgdp + '.csv', index=False)
    # Heatmap of scenario-specific breakdown
    fig_costmat_level = heatmap(
        input=costmat_level,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_level + '.png',
        title='WA, Kids & Elderly: Flat and ' + scenario_name,
        lb=0,
        ub=costmat_level.max().max(),
        format='.1f'
    )
    fig_costmat_percgdp = heatmap(
        input=costmat_percgdp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_percgdp + '.png',
        title='WA, Kids & Elderly: Flat and ' + scenario_name + '(% GDP)',
        lb=0,
        ub=costmat_percgdp.max().max(),
        format='.1f'
    )
    list_file_names = list_file_names + [file_name_level, file_name_percgdp]
    # Consolidate
    if round_scenario == 1:
        costmat_level_consol = pd.DataFrame(costmat_level['Total']).copy()
        costmat_percgdp_consol = pd.DataFrame(costmat_percgdp['Total']).copy()
    if round_scenario > 1:
        costmat_level_consol = pd.concat(
            [costmat_level_consol, costmat_level['Total']],
            axis=1)  # left-right concat
        costmat_percgdp_consol = pd.concat(
            [costmat_percgdp_consol, costmat_percgdp['Total']],
            axis=1)  # left-right concat
    # Distinguish new columns
    costmat_level_consol = costmat_level_consol.rename(columns={'Total': scenario_name})
    costmat_percgdp_consol = costmat_percgdp_consol.rename(columns={'Total': scenario_name})
    round_scenario += 1
# Heatmap for consolidated version
file_name_level = path_output + \
                  'cost_matrix_' + \
                  'kids_elderly_flat_restrict_' + \
                  'wa_flat_restrict' + \
                  '_level'
file_name_percgdp = path_output + \
                    'cost_matrix_' + \
                    'kids_elderly_flat_restrict_' + \
                    'wa_flat_restrict' + \
                    '_percgdp'
fig_costmat_level_consol = heatmap(
    input=costmat_level_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_level + '.png',
    title='WA, Kids & Elderly: Flat and Restricted',
    lb=0,
    ub=costmat_level_consol.max().max(),
    format='.1f'
)
fig_costmat_percgdp_consol = heatmap(
    input=costmat_percgdp_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_percgdp + '.png',
    title='WA, Kids & Elderly: Flat and Restricted (% GDP)',
    lb=0,
    ub=costmat_percgdp_consol.max().max(),
    format='.1f'
)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate parquet and csv files for consolidated disbursement table
costmat_level_consol.to_parquet(file_name_level + '.parquet')
costmat_level_consol.to_csv(file_name_level + '.csv', index=False)
costmat_percgdp_consol.to_parquet(file_name_percgdp + '.parquet')
costmat_percgdp_consol.to_csv(file_name_percgdp + '.csv', index=False)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate consolidated PDF
pil_img2pdf(list_images=list_file_names,
            extension='png',
            pdf_name=path_output + 'cost_matrix_' +
                     'kids_elderly_flat_restrict_' +
                     'wa_flat_restrict' +
                     '_consol')

# E --- Restricted and tier everything
round_scenario = 1
list_file_names = []
for scenario_name, choice_threshold_exclude_wa in tqdm(dict_scenarios_wa_restrict.items()):
    # Compute
    costmat_level, costmat_percgdp = compute_scenario_cost(
        multiplier_set=tiered_amount_multiplier,
        tier_wa=True,  # tiered rate
        tier_kids=True,
        tier_elderly=True,
        threshold_exclude_wa=choice_threshold_exclude_wa,  # restrict
        threshold_exclude_kids=choice_threshold_exclude_wa,
        threshold_exclude_elderly=choice_threshold_exclude_wa
    )
    # File names of scenario-specific breakdown
    file_name_level = path_output + \
                      'cost_matrix_' + \
                      'kids_elderly_tiered_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_' + \
                      'wa_tiered_restrict_' + \
                      str(choice_threshold_exclude_wa) + \
                      '_level'
    file_name_percgdp = path_output + \
                        'cost_matrix_' + \
                        'kids_elderly_tiered_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_' + \
                        'wa_tiered_restrict_' + \
                        str(choice_threshold_exclude_wa) + \
                        '_percgdp'
    # Export parquet and csv of scenario-specific breakdown
    costmat_level.to_parquet(file_name_level + '.parquet')
    costmat_level.to_csv(file_name_level + '.csv', index=False)
    costmat_percgdp.to_parquet(file_name_percgdp + '.parquet')
    costmat_percgdp.to_csv(file_name_percgdp + '.csv', index=False)
    # Heatmap of scenario-specific breakdown
    fig_costmat_level = heatmap(
        input=costmat_level,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_level + '.png',
        title='WA, Kids & Elderly: Tiered and ' + scenario_name,
        lb=0,
        ub=costmat_level.max().max(),
        format='.1f'
    )
    fig_costmat_percgdp = heatmap(
        input=costmat_percgdp,
        mask=False,
        colourmap='vlag',
        outputfile=file_name_percgdp + '.png',
        title='WA, Kids & Elderly: Tiered and ' + scenario_name + '(% GDP)',
        lb=0,
        ub=costmat_percgdp.max().max(),
        format='.1f'
    )
    list_file_names = list_file_names + [file_name_level, file_name_percgdp]
    # Consolidate
    if round_scenario == 1:
        costmat_level_consol = pd.DataFrame(costmat_level['Total']).copy()
        costmat_percgdp_consol = pd.DataFrame(costmat_percgdp['Total']).copy()
    if round_scenario > 1:
        costmat_level_consol = pd.concat(
            [costmat_level_consol, costmat_level['Total']],
            axis=1)  # left-right concat
        costmat_percgdp_consol = pd.concat(
            [costmat_percgdp_consol, costmat_percgdp['Total']],
            axis=1)  # left-right concat
    # Distinguish new columns
    costmat_level_consol = costmat_level_consol.rename(columns={'Total': scenario_name})
    costmat_percgdp_consol = costmat_percgdp_consol.rename(columns={'Total': scenario_name})
    round_scenario += 1
# Heatmap for consolidated version
file_name_level = path_output + \
                  'cost_matrix_' + \
                  'kids_elderly_tiered_restrict_' + \
                  'wa_tiered_restrict' + \
                  '_level'
file_name_percgdp = path_output + \
                    'cost_matrix_' + \
                    'kids_elderly_tiered_restrict_' + \
                    'wa_tiered_restrict' + \
                    '_percgdp'
fig_costmat_level_consol = heatmap(
    input=costmat_level_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_level + '.png',
    title='WA, Kids & Elderly: Tiered and Restricted',
    lb=0,
    ub=costmat_level_consol.max().max(),
    format='.1f'
)
fig_costmat_percgdp_consol = heatmap(
    input=costmat_percgdp_consol,
    mask=False,
    colourmap='vlag',
    outputfile=file_name_percgdp + '.png',
    title='WA, Kids & Elderly: Tiered and Restricted (% GDP)',
    lb=0,
    ub=costmat_percgdp_consol.max().max(),
    format='.1f'
)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate parquet and csv files for consolidated disbursement table
costmat_level_consol.to_parquet(file_name_level + '.parquet')
costmat_level_consol.to_csv(file_name_level + '.csv', index=False)
costmat_percgdp_consol.to_parquet(file_name_percgdp + '.parquet')
costmat_percgdp_consol.to_csv(file_name_percgdp + '.csv', index=False)
list_file_names = [file_name_level, file_name_percgdp] + list_file_names
# Generate consolidated PDF
pil_img2pdf(list_images=list_file_names,
            extension='png',
            pdf_name=path_output + 'cost_matrix_' +
                     'kids_elderly_tiered_restrict_' +
                     'wa_tiered_restrict' +
                     '_consol')

# X --- Export
telsendfiles(
    conf=tel_config,
    path=path_output + 'cost_matrix_' +
         'kids_elderly_flat_universal_' +
         'wa_tiered_restrict' +
         '_consol' + '.pdf',
    cap='cost_matrix_' +
        'kids_elderly_flat_universal_' +
        'wa_tiered_restrict' +
        '_consol'
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'cost_matrix_' +
         'kids_elderly_flat_universal_' +
         'wa_flat_restrict' +
         '_consol' + '.pdf',
    cap='cost_matrix_' +
        'kids_elderly_flat_universal_' +
        'wa_flat_restrict' +
        '_consol'
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'cost_matrix_' +
         'kids_elderly_flat_restrict_' +
         'wa_flat_restrict' +
         '_consol' + '.pdf',
    cap='cost_matrix_' +
        'kids_elderly_flat_restrict_' +
        'wa_flat_restrict' +
        '_consol'
)
telsendfiles(
    conf=tel_config,
    path=path_output + 'cost_matrix_' +
         'kids_elderly_tiered_restrict_' +
         'wa_tiered_restrict' +
         '_consol' + '.pdf',
    cap='cost_matrix_' +
        'kids_elderly_tiered_restrict_' +
        'wa_tiered_restrict' +
        '_consol'
)

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- analysis_descriptive_cost_matrix: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
