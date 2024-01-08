# %%
# Regression analysis, but overall, and by cohort quantiles
import pandas as pd
import numpy as np
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    fe_reg,
    re_reg,
    reg_ols,
    barchart,
    heatmap,
    pil_img2pdf,
)
from tabulate import tabulate
import dataframe_image as dfi
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
tel_config = os.getenv("TEL_CONFIG")
path_data = "data/hies_consol/"
income_choice = os.getenv("INCOME_CHOICE")
outcome_choice = os.getenv("OUTCOME_CHOICE")
use_first_diff = ast.literal_eval(os.getenv("USE_FIRST_DIFF"))
if use_first_diff:
    fd_suffix = "fd"
elif not use_first_diff:
    fd_suffix = "level"
show_ci = ast.literal_eval(os.getenv("SHOW_CI"))
hhbasis_adj_analysis = ast.literal_eval(os.getenv("HHBASIS_ADJ_ANALYSIS"))
equivalised_adj_analysis = ast.literal_eval(os.getenv("EQUIVALISED_ADJ_ANALYSIS"))
if hhbasis_adj_analysis:
    hhbasis_suffix = "_hhbasis"
    hhbasis_chart_title = " (Total HH)"
if equivalised_adj_analysis:
    hhbasis_suffix = "_equivalised"
    hhbasis_chart_title = " (Equivalised)"
elif not hhbasis_adj_analysis and not equivalised_adj_analysis:
    hhbasis_suffix = ""
    hhbasis_chart_title = ""
hhbasis_cohorts_with_hhsize = ast.literal_eval(os.getenv("HHBASIS_COHORTS_WITH_HHSIZE"))


# %%
# --------- Analysis Starts ---------


def load_clean_estimate(input_suffix, opt_income, opt_consumption, opt_first_diff):
    # I --- Load data
    df = pd.read_parquet(
        path_data
        + "hies_consol_agg_balanced_"
        + input_suffix
        + hhbasis_suffix
        + ".parquet"
    )

    # II --- Pre-analysis prep
    # Redefine year
    df = df.rename(columns={"_time": "year"})
    # Keep only entity + time + time-variant variables
    col_groups = [
        "state",
        "urban",
        "education",
        "ethnicity",
        "income_gen_members_group",
        "male",
        "birth_year_group",
        "marriage",
        "emp_status",
        "industry",
        "occupation",
    ]
    if hhbasis_adj_analysis and hhbasis_cohorts_with_hhsize:
        col_groups = col_groups + ["hh_size_group"]
    df[col_groups] = df[col_groups].astype("str")
    df["cohort_code"] = df[col_groups].sum(axis=1)
    df = df.drop(col_groups, axis=1)

    # logs
    col_cons = [
        "cons_01",
        "cons_02",
        "cons_03",
        "cons_04",
        "cons_05",
        "cons_06",
        "cons_07",
        "cons_08",
        "cons_09",
        "cons_10",
        "cons_11",
        "cons_12",
        "cons_13",
        "cons_01_12",
        "cons_01_13",
        "cons_0722_fuel",
        "cons_07_ex_bigticket",
    ]
    col_inc = [
        "salaried_wages",
        "other_wages",
        "asset_income",
        "gross_transfers",
        "gross_income",
    ]
    for i in col_cons + col_inc:
        pass
        # df[i] = np.log(df[i])

    # First diff
    if opt_first_diff:
        for y in col_cons + col_inc:
            # df[y] = (df[y] - df.groupby('cohort_code')[y].shift(1)) / df.groupby('cohort_code')[y].shift(1)
            df[y] = df[y] - df.groupby("cohort_code")[y].shift(1)
        df = df.dropna(axis=0)

    # III --- Estimation
    # Execute
    mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = fe_reg(
        df=df,
        y_col=opt_consumption,
        x_cols=[opt_income],
        i_col="cohort_code",
        t_col="year",
        fixed_effects=True,
        time_effects=False,
        cov_choice="robust",
    )

    (
        mod_timefe,
        res_timefe,
        params_table_timefe,
        joint_teststats_timefe,
        reg_det_timefe,
    ) = fe_reg(
        df=df,
        y_col=opt_consumption,
        x_cols=[opt_income],
        i_col="cohort_code",
        t_col="year",
        fixed_effects=False,
        time_effects=True,
        cov_choice="robust",
    )

    mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = re_reg(
        df=df,
        y_col=opt_consumption,
        x_cols=[opt_income],
        i_col="cohort_code",
        t_col="year",
        cov_choice="robust",
    )

    # IV --- Output
    return (
        params_table_fe,
        params_table_timefe,
        params_table_re,
        # joint_teststats_fe,
        # joint_teststats_timefe,
        # joint_teststats_re,
        reg_det_fe,
        reg_det_timefe,
        reg_det_re,
    )


# Loop to estimate all quantiles
list_quantiles = ["0-20", "20-40", "40-60", "60-80", "80-100"]
# [0.2, 0.4, 0.6, 0.8]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_suffixes = ["20p", "40p", "60p", "80p", "100p"]
# ['20p', '40p', '60p', '80p']  # ['10p', '20p', '30p', '40p', '50p', '60p', '70p', '80p', '90p']
round = 1
for quantile, suffix in tqdm(zip(list_quantiles, list_suffixes)):
    # Load, clean, and estimate
    (
        params_table_fe,
        params_table_timefe,
        params_table_re,
        # joint_teststats_fe,
        # joint_teststats_timefe,
        # joint_teststats_re,
        reg_det_fe,
        reg_det_timefe,
        reg_det_re,
    ) = load_clean_estimate(
        input_suffix=suffix,
        opt_income=income_choice,
        opt_consumption=outcome_choice,
        opt_first_diff=use_first_diff,
    )
    # Indicate outcome variable
    params_table_fe["outcome_variable"] = outcome_choice
    params_table_timefe["outcome_variable"] = outcome_choice
    params_table_re["outcome_variable"] = outcome_choice
    # Indicate quantile variable
    params_table_fe["quantile"] = quantile
    params_table_timefe["quantile"] = quantile
    params_table_re["quantile"] = quantile
    # Indicate method
    params_table_fe["method"] = "FE"
    params_table_timefe["method"] = "TimeFE"
    params_table_re["method"] = "RE"
    # Regression stats
    reg_det_fe.columns = [quantile]
    reg_det_timefe.columns = [quantile]
    reg_det_re.columns = [quantile]
    # Consolidate quantiles
    if round == 1:
        params_table_fe_consol = pd.DataFrame(
            params_table_fe.loc[income_choice]
        ).transpose()
        params_table_timefe_consol = pd.DataFrame(
            params_table_timefe.loc[income_choice]
        ).transpose()
        params_table_re_consol = pd.DataFrame(
            params_table_re.loc[income_choice]
        ).transpose()
        reg_det_fe_consol = reg_det_fe.copy()
        reg_det_timefe_consol = reg_det_timefe.copy()
        reg_det_re_consol = reg_det_re.copy()
    elif round >= 1:
        params_table_fe_consol = pd.concat(
            [
                params_table_fe_consol,
                pd.DataFrame(params_table_fe.loc[income_choice]).transpose(),
            ],
            axis=0,
        )
        params_table_timefe_consol = pd.concat(
            [
                params_table_timefe_consol,
                pd.DataFrame(params_table_timefe.loc[income_choice]).transpose(),
            ],
            axis=0,
        )
        params_table_re_consol = pd.concat(
            [
                params_table_re_consol,
                pd.DataFrame(params_table_re.loc[income_choice]).transpose(),
            ],
            axis=0,
        )
        reg_det_fe_consol = pd.concat([reg_det_fe_consol, reg_det_fe], axis=1)
        reg_det_timefe_consol = pd.concat(
            [reg_det_timefe_consol, reg_det_timefe], axis=1
        )
        reg_det_re_consol = pd.concat([reg_det_re_consol, reg_det_re], axis=1)
    round += 1

# Set type
dict_dtype = {
    "Parameter": "float",
    # 'SE': 'float',
    "LowerCI": "float",
    "UpperCI": "float",
    "quantile": "str",
}
params_table_fe_consol = params_table_fe_consol.astype(dict_dtype)
params_table_timefe_consol = params_table_timefe_consol.astype(dict_dtype)
params_table_re_consol = params_table_re_consol.astype(dict_dtype)
reg_det_fe_consol = reg_det_fe_consol.astype("float")
reg_det_timefe_consol = reg_det_timefe_consol.astype("float")
reg_det_re_consol = reg_det_re_consol.astype("float")

if not show_ci:
    for col in ["LowerCI", "UpperCI"]:
        del params_table_fe_consol[col]
        del params_table_timefe_consol[col]
        del params_table_re_consol[col]

# Export heat maps by methodology
list_variables_heatmaps = ["quantile", "Parameter"]
if show_ci:
    list_variables_heatmaps = list_variables_heatmaps + ["LowerCI", "UpperCI"]
list_heatmaps_file_names = []
# FE
d = params_table_fe_consol.copy()
d = d[list_variables_heatmaps].set_index("quantile")
file_name = (
    "output/params_table_overall_quantile_fe"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_fe = heatmap(
    input=d,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="FE: MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".2f",
)
file_name = (
    "output/reg_det_overall_quantile_fe"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_fe_regdet = heatmap(
    input=reg_det_fe_consol,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Reg Stats for FE:\n MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".4f",
)
# TimeFE
d = params_table_timefe_consol.copy()
d = d[list_variables_heatmaps].set_index("quantile")
file_name = (
    "output/params_table_overall_quantile_timefe"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_timefe = heatmap(
    input=d,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Time FE: MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".2f",
)
file_name = (
    "output/reg_det_overall_quantile_timefe"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_timefe_regdet = heatmap(
    input=reg_det_timefe_consol,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Reg Stats for Time FE:\n MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".4f",
)
# RE
d = params_table_re_consol.copy()
d = d[list_variables_heatmaps].set_index("quantile")
file_name = (
    "output/params_table_overall_quantile_re"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_re = heatmap(
    input=d,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="RE: MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".2f",
)
file_name = (
    "output/reg_det_overall_quantile_re"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
list_heatmaps_file_names = list_heatmaps_file_names + [file_name]
heatmap_re_regdet = heatmap(
    input=reg_det_re_consol,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Reg Stats for RE:\n MPC by income group" + hhbasis_chart_title,
    lb=0,
    ub=0.6,
    format=".4f",
)
# Compile and send
pdf_file_name = (
    "output/params_table_overall_quantile_consol"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
pil_img2pdf(
    list_images=list_heatmaps_file_names, extension="png", pdf_name=pdf_file_name
)
# telsendfiles(
#     conf=tel_config,
#     path=pdf_file_name + '.pdf',
#     cap=pdf_file_name
# )

# Mega merge
params_table_consol = pd.concat(
    [params_table_fe_consol, params_table_timefe_consol, params_table_re_consol], axis=0
)
# Order columns
if show_ci:
    params_table_consol = params_table_consol[
        ["outcome_variable", "method", "quantile", "Parameter", "LowerCI", "UpperCI"]
    ]
if not show_ci:
    params_table_consol = params_table_consol[
        ["outcome_variable", "method", "quantile", "Parameter"]
    ]
# Export as csv and image
params_table_consol.to_parquet(
    "output/params_table_overall_quantile"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)
params_table_consol.to_csv(
    "output/params_table_overall_quantile"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv"
)
# dfi.export(params_table_consol,
#            'output/params_table_overall_quantile' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix + '.png',
#            fontsize=3.8, dpi=800, table_conversion='chrome', chrome_path=None)  # to overcome mar2023 error
# telsendimg(
#     conf=tel_config,
#     path='output/params_table_overall_quantile' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix + '.png',
#     cap='params_table_overall_quantile' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix
# )
print(tabulate(params_table_consol.round(3), headers="keys", tablefmt="pretty"))

# Average all quantiles
if show_ci:
    params_table_consol_avg = (
        params_table_consol.groupby("method")[["Parameter", "LowerCI", "UpperCI"]]
        .mean(numeric_only=True)
        .reset_index()
    )
if not show_ci:
    params_table_consol_avg = (
        params_table_consol.groupby("method")[["Parameter"]]
        .mean(numeric_only=True)
        .reset_index()
    )
# dfi.export(params_table_consol_avg,
#            'output/params_table_overall_quantile_avg' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix + '.png')
# telsendimg(
#     conf=tel_config,
#     path='output/params_table_overall_quantile_avg' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix + '.png',
#     cap='params_table_overall_quantile_avg' + '_' + outcome_choice + '_' + income_choice + '_' + fd_suffix + hhbasis_suffix
# )
print(tabulate(params_table_consol_avg.round(3), headers="keys", tablefmt="pretty"))

# --------- Analysis Ends ---------

# %%
# X --- Notify
# telsendmsg(conf=tel_config,
#            msg='impact-household --- analysis_reg_overall_quantile: COMPLETED')

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
