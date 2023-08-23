# da best in da east

import pandas as pd
import numpy as np
import multiprocess as mp
from src.helper import telsendmsg, telsendimg, telsendfiles, fe_reg, re_reg, reg_ols, heatmap, pil_img2pdf, barchart
from datetime import date, timedelta
from tqdm import tqdm
import itertools
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/hies_consol/'
path_2014 = './data/hies_2014/'
path_2016 = './data/hies_2016/'
path_2019 = './data/hies_2019/'
income_choice = os.getenv('INCOME_CHOICE')
outcome_choice = os.getenv('OUTCOME_CHOICE')
use_first_diff = ast.literal_eval(os.getenv('USE_FIRST_DIFF'))
if use_first_diff:
    fd_suffix = 'fd'
elif not use_first_diff:
    fd_suffix = 'level'
hhbasis_adj_analysis = ast.literal_eval(os.getenv('HHBASIS_ADJ_ANALYSIS'))
equivalised_adj_analysis = ast.literal_eval(os.getenv('EQUIVALISED_ADJ_ANALYSIS'))
if hhbasis_adj_analysis:
    hhbasis_suffix = '_hhbasis'
    hhbasis_chart_title = ' (Total HH)'
if equivalised_adj_analysis:
    hhbasis_suffix = '_equivalised'
    hhbasis_chart_title = ' (Equivalised)'
elif not hhbasis_adj_analysis and not equivalised_adj_analysis:
    hhbasis_suffix = ''
    hhbasis_chart_title = ''

# I --- Define exhaustive list of cohorts
col_groups = \
    [
        'state',
        'urban',
        'education',
        'ethnicity',
        'malaysian',
        'income_gen_members_group',
        'male',
        'birth_year_group',
        'marriage',
        'emp_status',
        'industry',
        'occupation'
    ]
all_possible_cohorts = sum([list(map(list,
                                     itertools.combinations(col_groups, i))) for i in range(len(col_groups) + 1)],
                           [])
del all_possible_cohorts[0]  # delete blank

example_run = False
if example_run:
    all_possible_cohorts = all_possible_cohorts[-200:]


# II.A --- Define mega function for loading, wrangling, cleaning, and estimating


def load_clean_estimate(cohort):

    # ------------ Preliminaries necessary for 'spawn' (unnecessary if running on Unix)

    # Essential packages
    import pandas as pd
    from src.helper import fe_reg
    import os
    from dotenv import load_dotenv
    import ast

    # Prelims
    load_dotenv()
    path_2014 = './data/hies_2014/'
    path_2016 = './data/hies_2016/'
    path_2019 = './data/hies_2019/'
    income_choice = os.getenv('INCOME_CHOICE')
    outcome_choice = os.getenv('OUTCOME_CHOICE')
    use_first_diff = ast.literal_eval(os.getenv('USE_FIRST_DIFF'))
    if use_first_diff:
        fd_suffix = 'fd'
    elif not use_first_diff:
        fd_suffix = 'level'
    show_ci = ast.literal_eval(os.getenv('SHOW_CI'))
    hhbasis_adj_analysis = ast.literal_eval(os.getenv('HHBASIS_ADJ_ANALYSIS'))
    equivalised_adj_analysis = ast.literal_eval(os.getenv('EQUIVALISED_ADJ_ANALYSIS'))
    if hhbasis_adj_analysis:
        hhbasis_suffix = '_hhbasis'
    hhbasis_chart_title = ' (Total HH)'
    if equivalised_adj_analysis:
        hhbasis_suffix = '_equivalised'
        hhbasis_chart_title = ' (Equivalised)'
    elif not hhbasis_adj_analysis and not equivalised_adj_analysis:
        hhbasis_suffix = ''
        hhbasis_chart_title = ''

    # Load raw data
    df_14_withbase = pd.read_parquet(path_2014 + 'hies_2014_consol_trimmedoutliers_groupandbase.parquet')
    df_16_withbase = pd.read_parquet(path_2016 + 'hies_2016_consol_trimmedoutliers_groupandbase.parquet')
    df_19_withbase = pd.read_parquet(path_2019 + 'hies_2019_consol_trimmedoutliers_groupandbase.parquet')

    # Define interim within-loop functions
    def gen_pseudopanel(data1, data2, data3, list_cols_cohort):
        # Groupby operation
        df1_agg = data1.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
        df2_agg = data2.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
        df3_agg = data3.groupby(list_cols_cohort).mean(numeric_only=True).reset_index()
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
        # Output
        return df_agg_balanced

    # Generate cohort data
    df = gen_pseudopanel(
        data1=df_14_withbase,
        data2=df_16_withbase,
        data3=df_19_withbase,
        list_cols_cohort=cohort
    )

    # Redefine year
    df = df.rename(columns={'_time': 'year'})

    # Keep only SELECTED entity + time + time-variant variables
    df[cohort] = df[cohort].astype('str')
    df['cohort_code'] = df[cohort].sum(axis=1)
    df = df.drop(cohort, axis=1)

    # First diff
    col_cons = ['cons_01', 'cons_02', 'cons_03', 'cons_04',
                'cons_05', 'cons_06', 'cons_07', 'cons_08',
                'cons_09', 'cons_10', 'cons_11', 'cons_12',
                'cons_13',
                'cons_01_12', 'cons_01_13',
                'cons_0722_fuel', 'cons_07_ex_bigticket']
    col_inc = ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers', 'gross_income']
    if use_first_diff:
        for y in col_cons + col_inc:
            # df[y] = 100 * (df[y] - df.groupby('cohort_code')[y].shift(1)) / df.groupby('cohort_code')[y].shift(1)
            df[y] = 100 * (df[y] - df.groupby('cohort_code')[y].shift(1))
        df = df.dropna(axis=0)

    # ------------ The actual analysis

    # Base blank dataframe to hold parameter estimates
    params_table = pd.DataFrame(
        columns=['cohort_choice', 'Parameter', 'LowerCI', 'UpperCI']
    )  # so that paradoxical runs produce empty DF as output

    # Save params as string
    cohort_choice = ','.join(cohort)

    # Fixed effects only! (save computational needs)
    mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = \
        fe_reg(
            df=df,
            y_col=outcome_choice,
            x_cols=[income_choice],
            i_col='cohort_code',
            t_col='year',
            fixed_effects=True,
            time_effects=False,
            cov_choice='robust'
        )
    params_table_fe['cohort_choice'] = cohort_choice

    # Concat parameter estimates to base blank dataframe
    params_table = pd.concat([params_table, params_table_fe], axis=0)  # top-down

    # Print output
    print('Done: ' + cohort_choice)
    # Output
    return params_table


# II.B --- Run analysis with parallel processing

__file__ = 'workaround.py'
pool = mp.Pool()  # no arguments = fastest relative to nested loops in MWE
output = pool.map(load_clean_estimate, all_possible_cohorts)
params_table_consol = pd.concat(output)
params_table_consol = params_table_consol.reset_index(drop=True)  # unique indices per row

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
