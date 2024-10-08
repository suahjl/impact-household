# %%
# Regression analysis, but stratified, and by quantiles

import pandas as pd
import numpy as np
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    fe_reg,
    re_reg,
    reg_ols,
    heatmap,
    heatmap_layered,
    pil_img2pdf,
    barchart,
)
from tabulate import tabulate
from tqdm import tqdm
import dataframe_image as dfi
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
# --------- Analysis Starts (only cohort pseudo-panel data regressions) ---------


# %%
# Functions
def load_clean_estimate(input_suffix, opt_income, opt_first_diff, opt_show_ci):
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
        # df_ind[i] = np.log(df_ind[i])

    # First diff
    if opt_first_diff:
        for y in col_cons + col_inc:
            # df[y] = 100 * (df[y] - df.groupby('cohort_code')[y].shift(1)) / df.groupby('cohort_code')[y].shift(1)
            df[y] = 100 * (df[y] - df.groupby("cohort_code")[y].shift(1))
        df = df.dropna(axis=0)

    # III.A --- Estimation: stratify by outcomes (consumption categories)
    # Nice names for consumption categories
    dict_cons_nice = {
        "cons_01_12": "Consumption",
        "cons_01_13": "Consumption + Fin. Expenses",
        "cons_01": "Food & Beverages",
        "cons_02": "Alcohol & Tobacco",
        "cons_03": "Clothing & Footwear",
        "cons_04": "Rent & Utilities",
        "cons_05": "Furnishing, HH Equipment & Maintenance",
        "cons_06": "Healthcare",
        "cons_07": "Transport",
        "cons_08": "Communication",
        "cons_09": "Recreation & Culture",
        "cons_10": "Education",
        "cons_11": "Restaurant & Hotels",
        "cons_12": "Misc",
        "cons_13": "Financial Expenses",
        "cons_0722_fuel": "Transport: Fuel Only",
        "cons_07_ex_bigticket": "Transport ex. Vehicles & Maintenance",
    }
    # Define outcome lists
    list_outcome_choices = (
        ["cons_01_13", "cons_01_12"]
        + ["cons_0" + str(i) for i in range(1, 10)]
        + ["cons_1" + str(i) for i in range(0, 4)]
        + ["cons_0722_fuel", "cons_07_ex_bigticket"]
    )
    # Estimates: FE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = fe_reg(
            df=df,
            y_col=outcome,
            x_cols=[opt_income],
            i_col="cohort_code",
            t_col="year",
            fixed_effects=True,
            time_effects=False,
            cov_choice="robust",
        )
        params_table_fe["outcome_variable"] = outcome
        if round == 1:
            params_table_fe_consol = pd.DataFrame(
                params_table_fe.loc[opt_income]
            ).transpose()
        elif round >= 1:
            params_table_fe_consol = pd.concat(
                [
                    params_table_fe_consol,
                    pd.DataFrame(params_table_fe.loc[opt_income]).transpose(),
                ],
                axis=0,
            )
        round += 1
    params_table_fe_consol["outcome_variable"] = params_table_fe_consol[
        "outcome_variable"
    ].replace(dict_cons_nice)
    params_table_fe_consol = params_table_fe_consol.set_index("outcome_variable")
    params_table_fe_consol = params_table_fe_consol.astype("float")
    if not opt_show_ci:
        for col in ["LowerCI", "UpperCI"]:
            del params_table_fe_consol[col]

    # Estimates: time FE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        (
            mod_timefe,
            res_timefe,
            params_table_timefe,
            joint_teststats_timefe,
            reg_det_timefe,
        ) = fe_reg(
            df=df,
            y_col=outcome,
            x_cols=[opt_income],
            i_col="cohort_code",
            t_col="year",
            fixed_effects=False,
            time_effects=True,
            cov_choice="robust",
        )
        params_table_timefe["outcome_variable"] = outcome
        if round == 1:
            params_table_timefe_consol = pd.DataFrame(
                params_table_timefe.loc[opt_income]
            ).transpose()
        elif round >= 1:
            params_table_timefe_consol = pd.concat(
                [
                    params_table_timefe_consol,
                    pd.DataFrame(params_table_timefe.loc[opt_income]).transpose(),
                ],
                axis=0,
            )
        round += 1
    params_table_timefe_consol["outcome_variable"] = params_table_timefe_consol[
        "outcome_variable"
    ].replace(dict_cons_nice)
    params_table_timefe_consol = params_table_timefe_consol.set_index(
        "outcome_variable"
    )
    params_table_timefe_consol = params_table_timefe_consol.astype("float")
    if not opt_show_ci:
        for col in ["LowerCI", "UpperCI"]:
            del params_table_timefe_consol[col]

    # Estimates: RE
    round = 1
    for outcome in tqdm(list_outcome_choices):
        mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = re_reg(
            df=df,
            y_col=outcome,
            x_cols=[opt_income],
            i_col="cohort_code",
            t_col="year",
            cov_choice="robust",
        )
        params_table_re["outcome_variable"] = outcome
        if round == 1:
            params_table_re_consol = pd.DataFrame(
                params_table_re.loc[opt_income]
            ).transpose()
        elif round >= 1:
            params_table_re_consol = pd.concat(
                [
                    params_table_re_consol,
                    pd.DataFrame(params_table_re.loc[opt_income]).transpose(),
                ],
                axis=0,
            )
        round += 1
    params_table_re_consol["outcome_variable"] = params_table_re_consol[
        "outcome_variable"
    ].replace(dict_cons_nice)
    params_table_re_consol = params_table_re_consol.set_index("outcome_variable")
    params_table_re_consol = params_table_re_consol.astype("float")
    if not opt_show_ci:
        for col in ["LowerCI", "UpperCI"]:
            del params_table_re_consol[col]

    # IV --- Output
    return params_table_fe_consol, params_table_timefe_consol, params_table_re_consol


# %%
# Loop to estimate all quantiles
list_quantiles = ["0-20", "20-40", "40-60", "60-80", "80-100"]
# [0.2, 0.4, 0.6, 0.8]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_suffixes = ["20p", "40p", "60p", "80p", "100p"]
# ['20p', '40p', '60p', '80p']  # ['10p', '20p', '30p', '40p', '50p', '60p', '70p', '80p', '90p']
list_filenames_fe = []
list_filenames_timefe = []
list_filenames_re = []
round = 1
for quantile, suffix in tqdm(zip(list_quantiles, list_suffixes)):
    # Load, clean, and estimate
    params_table_fe, params_table_timefe, params_table_re = load_clean_estimate(
        input_suffix=suffix,
        opt_income=income_choice,
        opt_first_diff=use_first_diff,
        opt_show_ci=show_ci,
    )
    # Save quantile-method-specific frame as heatmap
    # FE
    heatmap_params_table_fe = heatmap(
        input=params_table_fe,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_fe_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
        + ".png",
        title="Fixed effects: MPC by cons type for "
        + quantile
        + " group"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_fe = list_filenames_fe + [
        "./output/params_table_fe_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
    ]
    # telsendimg(
    #     conf=tel_config,
    #     path='./output/params_table_fe_strat_cons_' +
    #          income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix + '.png',
    #     cap='params_table_fe_strat_cons_' +
    #         income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix
    # )
    # Time FE
    heatmap_params_table_timefe = heatmap(
        input=params_table_timefe,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_timefe_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
        + ".png",
        title="Time FE: MPC by cons type for "
        + quantile
        + " group"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_timefe = list_filenames_timefe + [
        "./output/params_table_timefe_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
    ]
    # telsendimg(
    #     conf=tel_config,
    #     path='./output/params_table_timefe_strat_cons_' +
    #          income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix + '.png',
    #     cap='params_table_timefe_strat_cons_' +
    #         income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix
    # )
    # RE
    heatmap_params_table_re = heatmap(
        input=params_table_re,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_re_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
        + ".png",
        title="Random Effects: MPC by cons type for "
        + quantile
        + " group"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_re = list_filenames_re + [
        "./output/params_table_re_strat_cons_"
        + income_choice
        + "_"
        + fd_suffix
        + "_"
        + suffix
        + hhbasis_suffix
    ]
    # telsendimg(
    #     conf=tel_config,
    #     path='./output/params_table_re_strat_cons_' +
    #          income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix + '.png',
    #     cap='params_table_re_strat_cons_' +
    #         income_choice + '_' + fd_suffix + '_' + suffix + hhbasis_suffix
    # )
    # Indicate quantile variable
    params_table_fe["quantile"] = quantile
    params_table_timefe["quantile"] = quantile
    params_table_re["quantile"] = quantile
    # Indicate method
    params_table_fe["method"] = "FE"
    params_table_timefe["method"] = "TimeFE"
    params_table_re["method"] = "RE"
    # Consolidate quantiles by methodology
    if round == 1:
        params_table_fe_consol = params_table_fe.copy()
        params_table_timefe_consol = params_table_timefe.copy()
        params_table_re_consol = params_table_re.copy()
    elif round >= 1:
        params_table_fe_consol = pd.concat(
            [params_table_fe_consol, params_table_fe.copy()], axis=0
        )
        params_table_timefe_consol = pd.concat(
            [params_table_timefe_consol, params_table_timefe.copy()], axis=0
        )
        params_table_re_consol = pd.concat(
            [params_table_re_consol, params_table_re.copy()], axis=0
        )
    round += 1

# Compile heatmaps into pdfs
pil_img2pdf(
    list_images=list_filenames_fe,
    extension="png",
    pdf_name="./output/params_table_fe_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_quantile"
    + hhbasis_suffix,
)
pil_img2pdf(
    list_images=list_filenames_timefe,
    extension="png",
    pdf_name="./output/params_table_timefe_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_quantile"
    + hhbasis_suffix,
)
pil_img2pdf(
    list_images=list_filenames_re,
    extension="png",
    pdf_name="./output/params_table_re_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_quantile"
    + hhbasis_suffix,
)
# for i in [
#     "./output/params_table_fe_strat_cons_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + "_quantile"
#     + hhbasis_suffix,
#     "./output/params_table_timefe_strat_cons_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + "_quantile"
#     + hhbasis_suffix,
#     "./output/params_table_re_strat_cons_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + "_quantile"
#     + hhbasis_suffix,
# ]:
#     telsendfiles(conf=tel_config, path=i + ".pdf", cap=i)

# %%
# Set type for consolidated dataframe
if show_ci:
    dict_dtype = {
        "Parameter": "float",
        # 'SE': 'float',
        "LowerCI": "float",
        "UpperCI": "float",
        "quantile": "str",
    }
if not show_ci:
    dict_dtype = {
        "Parameter": "float",
        # 'SE': 'float',
        "quantile": "str",
    }
params_table_fe_consol = params_table_fe_consol.astype(dict_dtype)
params_table_timefe_consol = params_table_timefe_consol.astype(dict_dtype)
params_table_re_consol = params_table_re_consol.astype(dict_dtype)

# %%
# Sort subgroups
if show_ci:
    col_sort_order = ["method", "quantile", "Parameter", "LowerCI", "UpperCI"]
if not show_ci:
    col_sort_order = ["method", "quantile", "Parameter"]
params_table_fe_consol = params_table_fe_consol[col_sort_order]
params_table_timefe_consol = params_table_timefe_consol[col_sort_order]
params_table_re_consol = params_table_re_consol[col_sort_order]

# %%
# Output full output
params_table_fe_consol.to_csv(
    "./output/params_table_fe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv"
)
params_table_fe_consol.to_parquet(
    "./output/params_table_fe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)

params_table_timefe_consol.to_csv(
    "./output/params_table_timefe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv"
)
params_table_timefe_consol.to_parquet(
    "./output/params_table_timefe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)

params_table_re_consol.to_csv(
    "./output/params_table_re_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv"
)
params_table_re_consol.to_parquet(
    "./output/params_table_re_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)

# %%
# Generate standard regression estimate table of MPCs (x-axis = income groups; y-axis = cons type)
params_table_fe_consol_wide_disp = params_table_fe_consol.copy()
params_table_fe_consol_wide_val = params_table_fe_consol_wide_disp[
    ["method", "quantile"] + ["Parameter"]
].copy()
params_table_fe_consol_wide_disp[["Parameter", "LowerCI", "UpperCI"]] = (
    params_table_fe_consol_wide_disp[["Parameter", "LowerCI", "UpperCI"]]
    .round(2)
    .astype("str")
)
params_table_fe_consol_wide_disp["Parameter"] = (
    params_table_fe_consol_wide_disp["Parameter"]
    + " \n ("
    + params_table_fe_consol_wide_disp["LowerCI"]
    + " to "
    + params_table_fe_consol_wide_disp["UpperCI"]
    + ")"
)
for col in ["LowerCI", "UpperCI"]:
    del params_table_fe_consol_wide_disp[col]
params_table_fe_consol_wide_disp = params_table_fe_consol_wide_disp.pivot(
    columns="quantile", values="Parameter"
)
params_table_fe_consol_wide_val = params_table_fe_consol_wide_val.pivot(
    columns="quantile", values="Parameter"
)
list_outcomes_nice = [
    "Consumption + Fin. Expenses",
    "Consumption",
    "Food & Beverages",
    "Alcohol & Tobacco",
    "Clothing & Footwear",
    "Rent & Utilities",
    "Furnishing, HH Equipment & Maintenance",
    "Healthcare",
    "Transport",
    "Communication",
    "Recreation & Culture",
    "Education",
    "Restaurant & Hotels",
    "Misc",
    "Financial Expenses",
    "Transport: Fuel Only",
    "Transport ex. Vehicles & Maintenance",
]
params_table_fe_consol_wide_disp = params_table_fe_consol_wide_disp.loc[list_outcomes_nice, :]
params_table_fe_consol_wide_val = params_table_fe_consol_wide_val.loc[list_outcomes_nice, :]
fig = heatmap_layered(
    actual_input=params_table_fe_consol_wide_val,
    disp_input=params_table_fe_consol_wide_disp,
    mask=False,
    colourmap="vlag",
    outputfile="./output/params_table_fe_strat_cons_"
    + "wide"
    + "_"
    + fd_suffix
    + "_quantile"
    + hhbasis_suffix,
    title="Fixed Effects: MPC by cons type" + hhbasis_chart_title,
    lb=-0.3,
    ub=0.3,
    format="s",
    annot_size=6,
    tick_sizes=6,
)


# %%
# Generate consumption-specific distribution of MPCs along income quantile groups (the opposite of earlier)
list_outcomes = (
    ["cons_01_13", "cons_01_12"]
    + ["cons_0" + str(i) for i in range(1, 10)]
    + ["cons_1" + str(i) for i in range(0, 4)]
    + ["cons_0722_fuel", "cons_07_ex_bigticket"]
)
list_outcomes_nice = [
    "Consumption + Fin. Expenses",
    "Consumption",
    "Food & Beverages",
    "Alcohol & Tobacco",
    "Clothing & Footwear",
    "Rent & Utilities",
    "Furnishing, HH Equipment & Maintenance",
    "Healthcare",
    "Transport",
    "Communication",
    "Recreation & Culture",
    "Education",
    "Restaurant & Hotels",
    "Misc",
    "Financial Expenses",
    "Transport: Fuel Only",
    "Transport ex. Vehicles & Maintenance",
]
list_filenames_fe = []
list_filenames_timefe = []
list_filenames_re = []
params_table_fe_consol = params_table_fe_consol.reset_index()
params_table_timefe_consol = params_table_timefe_consol.reset_index()
params_table_re_consol = params_table_re_consol.reset_index()
for outcome, outcome_nice in tqdm(zip(list_outcomes, list_outcomes_nice)):
    # Fixed Effects
    params_table_fe_cons = params_table_fe_consol[
        params_table_fe_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_fe_cons[col]
    params_table_fe_cons = params_table_fe_cons.sort_values(
        by="quantile", ascending=True
    )
    params_table_fe_cons = params_table_fe_cons.set_index("quantile")
    heatmap_params_table_fe_cons = heatmap(
        input=params_table_fe_cons,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_fe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png",
        title="FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_fe = list_filenames_fe + [
        "./output/params_table_fe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]
    params_table_fe_cons = params_table_fe_cons.reset_index()
    bar_params_table_fe_cons = barchart(
        data=params_table_fe_cons,
        y_col="Parameter",
        x_col="quantile",
        main_title="FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        decimal_points=2,
    )
    bar_params_table_fe_cons.write_image(
        "./output/bar_params_table_fe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png"
    )
    list_filenames_fe = list_filenames_fe + [
        "./output/bar_params_table_fe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]

    # Time Fixed Effects
    params_table_timefe_cons = params_table_timefe_consol[
        params_table_timefe_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_timefe_cons[col]
    params_table_timefe_cons = params_table_timefe_cons.sort_values(
        by="quantile", ascending=True
    )
    params_table_timefe_cons = params_table_timefe_cons.set_index("quantile")
    heatmap_params_table_timefe_cons = heatmap(
        input=params_table_timefe_cons,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_timefe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png",
        title="Time FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_timefe = list_filenames_timefe + [
        "./output/params_table_timefe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]
    params_table_timefe_cons = params_table_timefe_cons.reset_index()
    bar_params_table_timefe_cons = barchart(
        data=params_table_timefe_cons,
        y_col="Parameter",
        x_col="quantile",
        main_title="Time FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        decimal_points=2,
    )
    bar_params_table_timefe_cons.write_image(
        "./output/bar_params_table_timefe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png"
    )
    list_filenames_timefe = list_filenames_timefe + [
        "./output/bar_params_table_timefe_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]

    # Random Effects
    params_table_re_cons = params_table_re_consol[
        params_table_re_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_re_cons[col]
    params_table_re_cons = params_table_re_cons.sort_values(
        by="quantile", ascending=True
    )
    params_table_re_cons = params_table_re_cons.set_index("quantile")
    heatmap_params_table_re_cons = heatmap(
        input=params_table_re_cons,
        mask=False,
        colourmap="vlag",
        outputfile="./output/params_table_re_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png",
        title="RE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        lb=0,
        ub=0.6,
        format=".2f",
    )
    list_filenames_re = list_filenames_re + [
        "./output/params_table_re_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]
    params_table_re_cons = params_table_re_cons.reset_index()
    bar_params_table_re_cons = barchart(
        data=params_table_re_cons,
        y_col="Parameter",
        x_col="quantile",
        main_title="RE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by income quantile groups"
        + hhbasis_chart_title,
        decimal_points=2,
    )
    bar_params_table_re_cons.write_image(
        "./output/bar_params_table_re_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".png"
    )
    list_filenames_re = list_filenames_re + [
        "./output/bar_params_table_re_"
        + outcome
        + "_strat_incgroup_"
        + income_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
    ]

# %%
# Group charts of c-specific mpcs by ygroups into pdf by methodology
# FE
pil_img2pdf(
    list_images=list_filenames_fe,
    extension="png",
    pdf_name="./output/params_table_fe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix,
)
# Time FE
pil_img2pdf(
    list_images=list_filenames_timefe,
    extension="png",
    pdf_name="./output/params_table_timefe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix,
)
# RE
pil_img2pdf(
    list_images=list_filenames_re,
    extension="png",
    pdf_name="./output/params_table_re_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix,
)

# Send charts of c-specific mpcs by ygroups into pdf by methodology
# FE
# telsendfiles(
#     conf=tel_config,
#     path="./output/params_table_fe_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix
#     + ".pdf",
#     cap="params_table_fe_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix,
# )
# Time FE
# telsendfiles(
#     conf=tel_config,
#     path="./output/params_table_timefe_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix
#     + ".pdf",
#     cap="params_table_timefe_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix,
# )
# RE
# telsendfiles(
#     conf=tel_config,
#     path="./output/params_table_re_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix
#     + ".pdf",
#     cap="params_table_re_"
#     + "cons"
#     + "_strat_incgroup_"
#     + income_choice
#     + "_"
#     + fd_suffix
#     + hhbasis_suffix,
# )

# %%
# --------- Analysis Ends ---------

# %%
# X --- Notify
# telsendmsg(
#     conf=tel_config, msg="impact-household --- analysis_reg_strat_quantile: COMPLETED"
# )

# %%
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
