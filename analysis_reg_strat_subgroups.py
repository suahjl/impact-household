# Regression analysis, but stratified, and for subgroups

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

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv("TEL_CONFIG")
path_data = "data/hies_consol/"
path_subgroup = path_data + "subgroup/"
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


# --------- Analysis Starts (only cohort pseudo-panel data regressions) ---------


def load_clean_estimate(
    full_input_suffix, opt_income, opt_first_diff, opt_show_ci, use_balanced
):
    # I --- Load data
    if use_balanced:
        df = pd.read_parquet(
            path_subgroup + "hies_consol_agg_balanced_" + full_input_suffix + ".parquet"
        )
    elif not use_balanced:
        df = pd.read_parquet(
            path_subgroup + "hies_consol_agg_" + full_input_suffix + ".parquet"
        )

    # II --- Pre-analysis prep
    # Redefine year
    df = df.rename(columns={"_time": "year"})
    # Reduced set of col_groups (different from others)
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
    for outcome in list_outcome_choices:
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
    for outcome in list_outcome_choices:
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
    for outcome in list_outcome_choices:
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


# Part 1 --- Loop to estimate all scenarios
# To hold parameter estimates
list_filenames_fe = []
list_filenames_timefe = []
list_filenames_re = []
# Define subgroups; cross-check with process_consol-group
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_incgroups_nice = ["B20-", "B20+", "M20-", "M20+", "T20"]
list_benefit_objectives = ["kid", "working_age2", "elderly2"]
list_benefit_objectives_nice = ["Child", "Working Age", "Elderly"]
list_truncate_thresholds = [
    1,
    1,
    1,
]  # [2, 2, 2] [3, 2, 1] [3, 2, 2] [4, 3, 2] [2, 2, 1]
use_balanced_choice = False
round = 1
for incgroup_suffix, incgroup_nice in tqdm(
    zip(list_incgroups_suffix, list_incgroups_nice)
):
    for benefit_objective, benefit_objective_nice, truncate_threshold in zip(
        list_benefit_objectives, list_benefit_objectives_nice, list_truncate_thresholds
    ):
        # Input file suffixes
        for n_obj in range(1, truncate_threshold + 1):
            if n_obj < truncate_threshold:
                file_suffix = (
                    "mean_"
                    + incgroup_suffix
                    + "_"
                    + benefit_objective
                    + "_"
                    + str(n_obj)
                    + hhbasis_suffix
                )
                n_obj_nice = str(n_obj)
            elif n_obj == truncate_threshold:
                file_suffix = (
                    "mean_"
                    + incgroup_suffix
                    + "_"
                    + benefit_objective
                    + "_"
                    + str(n_obj)
                    + "plus"
                    + hhbasis_suffix
                )
                n_obj_nice = str(n_obj) + "+"
            elif n_obj > truncate_threshold:
                raise NotImplementedError("Check input file names")
            # Print combo
            print(
                "\n"
                + incgroup_nice
                + " with "
                + n_obj_nice
                + " "
                + benefit_objective_nice
            )
            # Load, clean, estimate
            params_table_fe, params_table_timefe, params_table_re = load_clean_estimate(
                full_input_suffix=file_suffix,
                opt_income=income_choice,
                opt_first_diff=use_first_diff,
                opt_show_ci=show_ci,
                use_balanced=use_balanced_choice,
            )
            # Save subgroup-method-specific frame as heatmap
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
                + file_suffix
                + ".png",
                title="Fixed effects: MPC by cons type for "
                + incgroup_nice
                + " with "
                + n_obj_nice
                + " "
                + benefit_objective_nice
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
                + file_suffix
            ]

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
                + file_suffix
                + ".png",
                title="Time FE: MPC by cons type for "
                + incgroup_nice
                + " with "
                + n_obj_nice
                + " "
                + benefit_objective_nice
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
                + file_suffix
            ]

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
                + file_suffix
                + ".png",
                title="Random effects: MPC by cons type for "
                + incgroup_nice
                + " with "
                + n_obj_nice
                + " "
                + benefit_objective_nice
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
                + file_suffix
            ]

            # Indicate income group
            params_table_fe["gross_income_group"] = incgroup_nice
            params_table_timefe["gross_income_group"] = incgroup_nice
            params_table_re["gross_income_group"] = incgroup_nice
            # Indicate benefit objective group
            params_table_fe["benefit_objective"] = benefit_objective_nice
            params_table_timefe["benefit_objective"] = benefit_objective_nice
            params_table_re["benefit_objective"] = benefit_objective_nice
            # Indicate number of benefit recipients per HH
            params_table_fe["benefit_recipients"] = n_obj_nice
            params_table_timefe["benefit_recipients"] = n_obj_nice
            params_table_re["benefit_recipients"] = n_obj_nice
            # Indicate method
            params_table_fe["method"] = "FE"
            params_table_timefe["method"] = "TimeFE"
            params_table_re["method"] = "RE"

            # Consolidate subgroups by methodology
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
    + "_subgroups"
    + hhbasis_suffix,
)
pil_img2pdf(
    list_images=list_filenames_timefe,
    extension="png",
    pdf_name="./output/params_table_timefe_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
)
pil_img2pdf(
    list_images=list_filenames_re,
    extension="png",
    pdf_name="./output/params_table_re_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
)
for i in [
    "./output/params_table_fe_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
    "./output/params_table_timefe_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
    "./output/params_table_re_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
]:
    telsendfiles(conf=tel_config, path=i + ".pdf", cap=i)

# Set type for consolidated dataframe
if show_ci:
    dict_dtype = {
        "Parameter": "float",
        # 'SE': 'float',
        "LowerCI": "float",
        "UpperCI": "float",
        "gross_income_group": "str",
        "benefit_objective": "str",
        "benefit_recipients": "str",
    }
if not show_ci:
    dict_dtype = {
        "Parameter": "float",
        # 'SE': 'float',
        "gross_income_group": "str",
        "benefit_objective": "str",
        "benefit_recipients": "str",
    }
params_table_fe_consol = params_table_fe_consol.astype(dict_dtype)
params_table_timefe_consol = params_table_timefe_consol.astype(dict_dtype)
params_table_re_consol = params_table_re_consol.astype(dict_dtype)
# Sort subgroups
if show_ci:
    col_sort_order = [
        "method",
        "gross_income_group",
        "benefit_objective",
        "benefit_recipients",
        "Parameter",
        "LowerCI",
        "UpperCI",
    ]
if not show_ci:
    col_sort_order = [
        "method",
        "gross_income_group",
        "benefit_objective",
        "benefit_recipients",
        "Parameter",
    ]
params_table_fe_consol = params_table_fe_consol[col_sort_order]
params_table_timefe_consol = params_table_timefe_consol[col_sort_order]
params_table_re_consol = params_table_re_consol[col_sort_order]

# Output full output
params_table_fe_consol.to_csv(
    "./output/params_table_fe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".csv"
)
params_table_fe_consol.to_parquet(
    "./output/params_table_fe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".parquet"
)

params_table_timefe_consol.to_csv(
    "./output/params_table_timefe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".csv"
)
params_table_timefe_consol.to_parquet(
    "./output/params_table_timefe_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".parquet"
)

params_table_re_consol.to_csv(
    "./output/params_table_re_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".csv"
)
params_table_re_consol.to_parquet(
    "./output/params_table_re_consol_strat_cons_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".parquet"
)

# Part 2 --- Generate heatmaps and bar charts for consumption-specific MPCs by subgroups
# Set up parameters
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
# List to generate pdf files
list_filenames_fe = []
list_filenames_timefe = []
list_filenames_re = []
# Reset index of full output data frames
params_table_fe_consol = params_table_fe_consol.reset_index()
params_table_timefe_consol = params_table_timefe_consol.reset_index()
params_table_re_consol = params_table_re_consol.reset_index()
# Generate subgroup column
params_table_fe_consol["subgroup"] = (
    params_table_fe_consol["gross_income_group"]
    + " with "
    + params_table_fe_consol["benefit_recipients"]
    + " "
    + params_table_fe_consol["benefit_objective"]
)
params_table_timefe_consol["subgroup"] = (
    params_table_timefe_consol["gross_income_group"]
    + " with "
    + params_table_timefe_consol["benefit_recipients"]
    + " "
    + params_table_timefe_consol["benefit_objective"]
)
params_table_re_consol["subgroup"] = (
    params_table_re_consol["gross_income_group"]
    + " with "
    + params_table_re_consol["benefit_recipients"]
    + " "
    + params_table_re_consol["benefit_objective"]
)
# Remove disaggregated subgroup columns
for i in ["gross_income_group", "benefit_recipients", "benefit_objective"]:
    del params_table_fe_consol[i]
    del params_table_timefe_consol[i]
    del params_table_re_consol[i]
# Loop to generate heatmaps and bar charts
for outcome, outcome_nice in tqdm(zip(list_outcomes, list_outcomes_nice)):
    # Fixed Effects
    params_table_fe_cons = params_table_fe_consol[
        params_table_fe_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_fe_cons[col]
    params_table_fe_cons = params_table_fe_cons.set_index("subgroup")
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
        + "_subgroups"
        + hhbasis_suffix
        + ".png",
        title="FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]
    params_table_fe_cons = params_table_fe_cons.reset_index()
    bar_params_table_fe_cons = barchart(
        data=params_table_fe_cons,
        y_col="Parameter",
        x_col="subgroup",
        main_title="FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]

    # Time Fixed Effects
    params_table_timefe_cons = params_table_timefe_consol[
        params_table_timefe_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_timefe_cons[col]
    # params_table_timefe_cons = params_table_timefe_cons.sort_values(by='subgroup', ascending=True)
    params_table_timefe_cons = params_table_timefe_cons.set_index("subgroup")
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
        + "_subgroups"
        + hhbasis_suffix
        + ".png",
        title="Time FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]
    params_table_timefe_cons = params_table_timefe_cons.reset_index()
    bar_params_table_timefe_cons = barchart(
        data=params_table_timefe_cons,
        y_col="Parameter",
        x_col="subgroup",
        main_title="Time FE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]

    # Random Effects
    params_table_re_cons = params_table_re_consol[
        params_table_re_consol["outcome_variable"] == outcome_nice
    ].copy()
    for col in ["outcome_variable", "method"]:
        del params_table_re_cons[col]
    # params_table_re_cons = params_table_re_cons.sort_values(by='subgroup', ascending=True)
    params_table_re_cons = params_table_re_cons.set_index("subgroup")
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
        + "_subgroups"
        + hhbasis_suffix
        + ".png",
        title="RE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]
    params_table_re_cons = params_table_re_cons.reset_index()
    bar_params_table_re_cons = barchart(
        data=params_table_re_cons,
        y_col="Parameter",
        x_col="subgroup",
        main_title="RE: MPC of "
        + outcome_nice
        + " from "
        + income_choice
        + " by Subgroups"
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
        + "_subgroups"
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
        + "_subgroups"
        + hhbasis_suffix
    ]

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
    + "_subgroups"
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
    + "_subgroups"
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
    + "_subgroups"
    + hhbasis_suffix,
)

# Send charts of c-specific mpcs by ygroups into pdf by methodology
# FE
telsendfiles(
    conf=tel_config,
    path="./output/params_table_fe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".pdf",
    cap="params_table_fe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
)
# Time FE
telsendfiles(
    conf=tel_config,
    path="./output/params_table_timefe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".pdf",
    cap="params_table_timefe_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
)
# RE
telsendfiles(
    conf=tel_config,
    path="./output/params_table_re_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix
    + ".pdf",
    cap="params_table_re_"
    + "cons"
    + "_strat_incgroup_"
    + income_choice
    + "_"
    + fd_suffix
    + "_subgroups"
    + hhbasis_suffix,
)

# --------- Analysis Ends ---------

# X --- Notify
telsendmsg(
    conf=tel_config, msg="impact-household --- analysis_reg_strat_subgroups: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
