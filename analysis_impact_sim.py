# Use income group-specific mpc to simulate total group impact

import pandas as pd
import numpy as np
from tabulate import tabulate
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    heatmap,
    pil_img2pdf,
    wide_grouped_barchart,
)
import dataframe_image as dfi
from datetime import timedelta, date
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# 0 --- Main settings
# Paths and script-specific settings
load_dotenv()
choice_horizon = int(os.getenv("IRF_HORIZON"))
tel_config = os.getenv("TEL_CONFIG")
path_output = "output/"

# Which macro-analysis
macro_qoq = ast.literal_eval(os.getenv("MACRO_QOQ"))
macro_yoy = ast.literal_eval(os.getenv("MACRO_YOY"))
macro_ln_levels = ast.literal_eval(os.getenv("MACRO_LN_LEVELS"))
macro_ln_qoq = ast.literal_eval(os.getenv("MACRO_LN_QOQ"))
macro_ln_yoy = ast.literal_eval(os.getenv("MACRO_LN_YOY"))
if (
    not macro_qoq
    and not macro_yoy
    and not macro_ln_levels
    and not macro_ln_qoq
    and not macro_ln_yoy
):
    macro_suffix = "levels"
if (
    macro_qoq
    and not macro_yoy
    and not macro_ln_levels
    and not macro_ln_qoq
    and not macro_ln_yoy
):
    macro_suffix = "qoq"
if (
    macro_yoy
    and not macro_qoq
    and not macro_ln_levels
    and not macro_ln_qoq
    and not macro_ln_yoy
):
    macro_suffix = "yoy"
if (
    macro_ln_levels
    and not macro_yoy
    and not macro_qoq
    and not macro_ln_qoq
    and not macro_ln_yoy
):
    macro_suffix = "ln_levels"
if (
    macro_ln_qoq
    and not macro_qoq
    and not macro_yoy
    and not macro_ln_levels
    and not macro_ln_yoy
):
    macro_suffix = "ln_qoq"
if (
    macro_ln_yoy
    and not macro_qoq
    and not macro_yoy
    and not macro_ln_levels
    and not macro_ln_qoq
):
    macro_suffix = "ln_yoy"

# Which micro-analysis
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

# Which parameters
choice_macro_shock = "Private Consumption"
choice_micro_outcome = "Consumption"
choices_macro_response = [
    "GDP",
    "Private Consumption",
    "Investment",
    "CPI",
    "NEER",
    "MGS 10-Year Yields",
]
choice_rounds_to_repeat = 1
choice_max_q = 7

# I --- Load MPC, IRF, and disbursement estimates
# IRF estimates
irf = pd.read_parquet(
    path_output + "macro_var_irf_all_varyx_narrative_avg_" + macro_suffix + ".parquet"
)
# Nice names
dict_cols_nice = {
    "ex": "Exports",
    "gc": "Public Consumption",
    "pc": "Private Consumption",
    "gfcf": "Investment",
    "im": "Imports",
    "gdp": "GDP",
    "brent": "Brent",
    "cpi": "CPI",
    "gepu": "Uncertainty",
    "myrusd": "MYR/USD",
    "mgs10y": "MGS 10-Year Yields",
    "klibor1m": "KLIBOR 1-Month",
    "neer": "NEER",
    "reer": "REER",
    "klibor3m": "KLIBOR 3-Month",
    "importworldipi": "Import-Weighted World IPI",
    "prodworldipi": "Production-Weighted World IPI",
    "maxgepu": "Uncertainty Shocks",
}
for i in ["shock", "response"]:
    irf[i] = irf[i].replace(dict_cols_nice)
# Restrict IRFs to shock of choice
irf = irf[irf["shock"] == choice_macro_shock]
# MPC estimates
mpc = pd.read_parquet(
    path_output
    + "params_table_overall_quantile"
    + "_"
    + outcome_choice
    + "_"
    + income_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
).reset_index()
mpc = mpc[mpc["method"] == "FE"]
mpc = mpc[["quantile", "Parameter"]]
mpc = mpc.rename(columns={"quantile": "gross_income_group"})
mpc["gross_income_group"] = mpc["gross_income_group"].replace(
    {"0-20": "B20-", "20-40": "B20+", "40-60": "M20-", "60-80": "M20+", "80-100": "T20"}
)

# III.0 --- Set base parameters
list_incgroups = ["B20-", "B20+", "M20-", "M20+", "T20"]
ngdp_2022 = 1788184000000
rgdp_2022 = 1507305000000
npc_2022 = 1029952000000
rpc_2022 = 907573000000
deflator_2022 = 118.65
deflator_2023 = 118.65
pc_deflator_2022 = 113.48
pc_deflator_2023 = 113.48

pc_ltavg = 6.934
gfcf_ltavg = 2.450
gdp_ltavg = 4.951
neer_ltavg = -2.436
cpi_ltavg = 1.925
mgs10y_ltavg = -0.08
dict_ltavg = {
    "GDP": gdp_ltavg,
    "Private Consumption": pc_ltavg,
    "Investment": gfcf_ltavg,
    "CPI": cpi_ltavg,
    "NEER": neer_ltavg,
    "MGS 10-Year Yields": mgs10y_ltavg,
}


# III --- Compute landing impact using estimated MPCs
# Function
def landing_impact_matrix(mpc, disb, incgroups):
    # Prelims
    mpc = mpc.copy()
    df_disb = disb.copy()
    # Trim total column for convenience
    df_disb = df_disb[incgroups]
    df_landing = df_disb.copy()
    # Compute c = g * mpc
    for incgroup in incgroups:
        df_landing[incgroup] = (
            df_landing[incgroup]
            * mpc.loc[mpc["gross_income_group"] == incgroup, "Parameter"].max()
        )
    # Output
    return df_landing


# Compute + beautify + output all landing impact matrices + STORE ORDERED LIST OF LANDING IMPACT
ordered_list_landing_impact_quantum_rpc = []
ordered_list_landing_impact_quantum_rgdp = []
files_landing_nominal = []
files_landing_rpc = []
files_landing_rgdp = []
for tiering, tiering_nice in zip(["tiered", "flat"], ["Tiered", "Flat"]):
    for restrict_threshold in tqdm([4, 5, False]):
        # Load --- RESTRICTIONS ON ALL, TIERED / FLAT ON ALL
        disb = pd.read_parquet(
            path_output
            + "cost_matrix_"
            + "kids_elderly_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_level"
            + ".parquet"
        )

        # Compute full landing impact
        landing = landing_impact_matrix(mpc=mpc, disb=disb, incgroups=list_incgroups)

        # Calculate total impact by row
        landing["Landing Impact"] = landing.sum(axis=1)

        # Convert into real PC growth
        landing_rpc = 100 * (
            (landing * 1000000000 / (pc_deflator_2023 / 100)) / rpc_2022
        )

        # Convert into real GDP growth
        landing_rgdp = 100 * (
            (landing * 1000000000 / (deflator_2023 / 100)) / rgdp_2022
        )

        # Convert landing impact into heat maps
        # Nominal
        file_name = (
            path_output
            + "landing_impact_sim_"
            + "kids_elderly"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_level"
            + "_nominal_"
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        fig_landing_nominal = heatmap(
            input=landing,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title="Landing Impact (RM bil): " + tiering_nice + " & Restricted",
            lb=0,
            ub=landing.max().max(),
            format=".1f",
        )
        files_landing_nominal = files_landing_nominal + [file_name]
        # PC growth
        file_name = (
            path_output
            + "landing_impact_sim_"
            + "kids_elderly"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_level"
            + "_rpc_"
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        fig_landing_rpc = heatmap(
            input=landing_rpc,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title="Landing Impact (PC; pp): " + tiering_nice + " & Restricted",
            lb=0,
            ub=landing_rpc.max().max(),
            format=".1f",
        )
        files_landing_rpc = files_landing_rpc + [file_name]

        # GDP growth
        file_name = (
            path_output
            + "landing_impact_sim_"
            + "kids_elderly"
            + tiering  # flat or tiered
            + "_restrict_"  # restrict ONLY
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering  # flat or tiered
            + "_restrict_"  # restrict ONLY
            + str(restrict_threshold)
            + "_level"
            + "_rgdp_"
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        fig_landing_rgdp = heatmap(
            input=landing_rgdp,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title="Landing Impact (GDP; pp): " + tiering_nice + " & Restricted",
            lb=0,
            ub=landing_rgdp.max().max(),
            format=".1f",
        )
        files_landing_rgdp = files_landing_rgdp + [file_name]
        # ADD TO ORDERED LIST OF LANDING IMPACT (RPC GROWTH PP, and GDP GROWTH PP)
        # PC
        landing_impact_quantum_rpc = landing_rpc.loc[
            "All Benefits", "Landing Impact"
        ].max()
        ordered_list_landing_impact_quantum_rpc = (
            ordered_list_landing_impact_quantum_rpc + [landing_impact_quantum_rpc]
        )
        # GDP
        landing_impact_quantum_rgdp = landing_rgdp.loc[
            "All Benefits", "Landing Impact"
        ].max()
        ordered_list_landing_impact_quantum_rgdp = (
            ordered_list_landing_impact_quantum_rgdp + [landing_impact_quantum_rgdp]
        )
# Compile landing impact heatmaps
pdf_name = (
    path_output
    + "landing_impact_sim_"
    + "_nominal_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
pil_img2pdf(list_images=files_landing_nominal, extension="png", pdf_name=pdf_name)
telsendfiles(conf=tel_config, path=pdf_name + ".pdf", cap=pdf_name)

pdf_name = (
    path_output
    + "landing_impact_sim_"
    + "rpc_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
pil_img2pdf(list_images=files_landing_rpc, extension="png", pdf_name=pdf_name)
telsendfiles(conf=tel_config, path=pdf_name + ".pdf", cap=pdf_name)

pdf_name = (
    path_output
    + "landing_impact_sim_"
    + "rgdp_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
)
pil_img2pdf(list_images=files_landing_rgdp, extension="png", pdf_name=pdf_name)
telsendfiles(conf=tel_config, path=pdf_name + ".pdf", cap=pdf_name)


# IV --- Use landing impact to compute indirect impact using estimated OIRFs from VAR + Compute aggregated impact
# Function
def compute_var_indirect_impact(
    irf, list_responses, shock, shock_size, convert_q_to_a, max_q
):
    # deep copy of parsed IRF
    indirect = irf.copy()
    # parse further
    indirect = indirect[
        indirect["response"].isin(list_responses) & indirect["shock"].isin([shock])
    ]
    # scale IRFs (originally unit shock)
    indirect["irf"] = indirect["irf"] * shock_size
    # limit quarters
    indirect = indirect[indirect["horizon"] <= max_q]
    # reset index
    indirect = indirect.reset_index(drop=True)
    # convert from quarterly response to annual response
    if convert_q_to_a:
        indirect["horizon_year"] = indirect["horizon"] // 4
        indirect = (
            indirect.groupby(["shock", "response", "horizon_year"])["irf"]
            .mean()
            .reset_index()
        )
        # Clean up
        indirect = indirect[indirect["horizon_year"] < 4]  # ignore final quarter
    # Clean up
    indirect = indirect.rename(columns={"irf": "indirect_impact"})
    indirect_rounded = indirect.copy()
    indirect_rounded["indirect_impact"] = indirect_rounded["indirect_impact"].round(2)
    # output
    return indirect, indirect_rounded


def create_aggregate_impact(indirect, landing_shock_rpc, landing_shock_rgdp):
    # Deep copy
    aggregate = indirect.copy()
    # Input landing impact (real PC and real GDP growth pp)
    aggregate.loc[
        (
            (aggregate["horizon_year"] == 0)
            & (aggregate["response"] == "Private Consumption")
        ),
        "landing_impact",
    ] = landing_shock_rpc
    aggregate.loc[
        ((aggregate["horizon_year"] == 0) & (aggregate["response"] == "GDP")),
        "landing_impact",
    ] = landing_shock_rgdp
    # Compute total impact across time
    aggregate["landing_impact"] = aggregate["landing_impact"].fillna(0)
    aggregate["total_impact"] = (
        aggregate["indirect_impact"] + aggregate["landing_impact"]
    )
    # Clean up
    aggregate = aggregate[
        [
            "shock",
            "response",
            "horizon_year",
            "landing_impact",
            "indirect_impact",
            "total_impact",
        ]
    ]
    aggregate_rounded = aggregate.copy()
    aggregate_rounded[
        ["landing_impact", "indirect_impact", "total_impact"]
    ] = aggregate_rounded[["landing_impact", "indirect_impact", "total_impact"]].round(
        2
    )
    # Output
    return aggregate, aggregate_rounded


def repeated_aggregate_impact(
    aggregate, landing_shock_rpc, landing_shock_rgdp, indirect, rounds_to_repeat
):
    # Deep copy
    repeated_agg = aggregate.copy()
    # Repeat landing impact beyond year 0 for (PC and GDP)
    repeated_agg.loc[
        (
            (repeated_agg["horizon_year"] <= rounds_to_repeat)
            & (repeated_agg["shock"] == "Private Consumption")
            & (repeated_agg["response"] == "Private Consumption")
        ),
        "landing_impact",
    ] = landing_shock_rpc
    repeated_agg.loc[
        (
            (repeated_agg["horizon_year"] <= rounds_to_repeat)
            & (repeated_agg["shock"] == "Private Consumption")
            & (repeated_agg["response"] == "GDP")
        ),
        "landing_impact",
    ] = landing_shock_rgdp
    # Repeat indirect impact beyond year 0
    round = 1
    while round <= rounds_to_repeat:
        repeated_agg.loc[
            repeated_agg["horizon_year"] >= round, "indirect_impact"
        ] = repeated_agg["indirect_impact"] + indirect.groupby(["shock", "response"])[
            "indirect_impact"
        ].shift(
            round
        )
        round += 1
    # Recalculate total impact
    repeated_agg["total_impact"] = (
        repeated_agg["landing_impact"] + repeated_agg["indirect_impact"]
    )
    # Clean up
    repeated_agg_rounded = repeated_agg.copy()
    repeated_agg_rounded[
        ["landing_impact", "indirect_impact", "total_impact"]
    ] = repeated_agg_rounded[
        ["landing_impact", "indirect_impact", "total_impact"]
    ].round(
        2
    )
    # Output
    return repeated_agg, repeated_agg_rounded


def export_dfi_parquet_csv_telegram(input, file_name):
    input.to_parquet(file_name + ".parquet")
    input.to_csv(file_name + ".csv", index=False)
    # dfi.export(input, file_name + '.png',
    #            fontsize=1.5, dpi=1600, table_conversion='chrome', chrome_path=None)
    # telsendimg(
    #     conf=tel_config,
    #     path=file_name + '.png',
    #     cap=file_name
    # )


# Compute all indirect impact matrices + aggregated + repeat aggregated
landing_impact_tracker = 0
list_scenarios_names_total_impact = []
list_scenarios_names_impact_breakdown = []
for tiering, tiering_nice in zip(["tiered", "flat"], ["Tiered", "Flat"]):
    for restrict_threshold, restrict_threshold_nice in tqdm(
        zip([4, 5, False], ["B60", "B80", "All"])
    ):
        # 0. Extract landing impact size
        landing_impact_quantum_rpc = ordered_list_landing_impact_quantum_rpc[
            landing_impact_tracker
        ]
        landing_impact_quantum_rgdp = ordered_list_landing_impact_quantum_rgdp[
            landing_impact_tracker
        ]

        # A. Indirect impact
        # Compute
        indirect, indirect_rounded = compute_var_indirect_impact(
            irf=irf,
            list_responses=choices_macro_response,
            shock=choice_macro_shock,
            shock_size=landing_impact_quantum_rpc,
            convert_q_to_a=True,
            max_q=choice_max_q,
        )
        # Output
        file_name = (
            path_output
            + "indirect_impact_sim_"
            + "kids_elderly"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        export_dfi_parquet_csv_telegram(input=indirect_rounded, file_name=file_name)

        # B. Aggregated impact
        # Compute
        aggregate, aggregate_rounded = create_aggregate_impact(
            indirect=indirect,
            landing_shock_rpc=landing_impact_quantum_rpc,
            landing_shock_rgdp=landing_impact_quantum_rgdp,
        )
        # Output
        file_name = (
            path_output
            + "aggregate_impact_sim_"
            + "kids_elderly"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        export_dfi_parquet_csv_telegram(input=aggregate_rounded, file_name=file_name)

        # C. Repeated aggregate impact
        # Compute
        repeated_agg, repeated_agg_rounded = repeated_aggregate_impact(
            aggregate=aggregate,
            landing_shock_rpc=landing_impact_quantum_rpc,
            landing_shock_rgdp=landing_impact_quantum_rgdp,
            indirect=indirect,
            rounds_to_repeat=choice_rounds_to_repeat,
        )
        # Output
        file_name = (
            path_output
            + "repeated_agg_impact_sim"
            + "kids_elderly"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + "_"
            + "wa_"
            + tiering
            + "_restrict_"
            + str(restrict_threshold)
            + income_choice
            + "_"
            + outcome_choice
            + "_"
            + fd_suffix
            + hhbasis_suffix
        )
        export_dfi_parquet_csv_telegram(input=repeated_agg_rounded, file_name=file_name)

        # D. Add to aggregated frame
        repeated_agg = repeated_agg.rename(
            columns={
                "landing_impact": tiering_nice
                + " and "
                + restrict_threshold_nice
                + ": Landing",
                "indirect_impact": tiering_nice
                + " and "
                + restrict_threshold_nice
                + ": Indirect",
                "total_impact": tiering_nice
                + " and "
                + restrict_threshold_nice
                + ": Total",
            }
        )
        if landing_impact_tracker == 0:
            allcombos = repeated_agg.copy()
        elif landing_impact_tracker > 0:
            allcombos = allcombos.merge(
                repeated_agg, how="left", on=["shock", "response", "horizon_year"]
            )

        # X. Move to next in the list of landing impact + extend scenarios list
        landing_impact_tracker += 1
        list_scenarios_names_total_impact = list_scenarios_names_total_impact + [
            tiering_nice + " and " + restrict_threshold_nice + ": Total"
        ]
        list_scenarios_names_impact_breakdown = (
            list_scenarios_names_impact_breakdown
            + [
                tiering_nice + " and " + restrict_threshold_nice + ": Landing",
                tiering_nice + " and " + restrict_threshold_nice + ": Indirect",
                tiering_nice + " and " + restrict_threshold_nice + ": Total",
            ]
        )
# Output allcombos
allcombos.to_parquet(
    path_output
    + "allcombos_consol_"
    + "shock_response_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)
allcombos.to_csv(
    path_output
    + "allcombos_consol_"
    + "shock_response_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv",
    index=False,
)


# V --- Compute levels impact from repeated aggregate impact of all combos
def compute_levels_impact(
    allcombos, list_impact_cols, list_shocks, list_responses, dict_ltavg
):
    round_shock_response = 0
    # Cycle through all shock-response combos
    for shock in list_shocks:
        for response in tqdm(list_responses):
            # Parse shock-response combo
            allcombos_sub = allcombos[
                (allcombos["shock"] == shock) & (allcombos["response"] == response)
            ].copy()

            # Create data frame to house levels
            allcombos_sub_cf = allcombos_sub.copy()
            allcombos_sub_level = allcombos_sub.copy()

            # Compute counterfactual levels using LT avg
            ltavg = dict_ltavg[response]
            for horizon in range(0, len(allcombos_sub_cf)):
                allcombos_sub_cf.loc[
                    allcombos_sub_cf["horizon_year"] == horizon, list_impact_cols
                ] = 100 * (1 + ltavg / 100) ** (horizon + 1)

            # Compute realised levels
            round_sub_levels = 1
            for horizon in range(0, len(allcombos_sub_cf)):
                if round_sub_levels == 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level["horizon_year"] == horizon, list_impact_cols
                    ] = 100 * (
                        1
                        + (
                            ltavg
                            + allcombos_sub.loc[
                                allcombos_sub["horizon_year"] == horizon,
                                list_impact_cols,
                            ]
                        )
                        / 100
                    )
                elif round_sub_levels > 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level["horizon_year"] == horizon, list_impact_cols
                    ] = allcombos_sub_level[list_impact_cols].shift(1) * (
                        1
                        + (
                            ltavg
                            + allcombos_sub.loc[
                                allcombos_sub["horizon_year"] == horizon,
                                list_impact_cols,
                            ]
                        )
                        / 100
                    )
                round_sub_levels += 1

            # Compute levels impact
            allcombos_sub_level_gap = allcombos_sub_level.copy()
            allcombos_sub_level_gap[list_impact_cols] = (
                allcombos_sub_level_gap[list_impact_cols]
                - allcombos_sub_cf[list_impact_cols]
            )

            # Consolidate
            if round_shock_response == 0:
                allcombos_level_gap = allcombos_sub_level_gap.copy()
            elif round_shock_response > 0:
                allcombos_level_gap = pd.concat(
                    [allcombos_level_gap, allcombos_sub_level_gap], axis=0
                )  # top-down

            # Move to next combo
            round_shock_response += 1

    # Output
    return allcombos_level_gap


allcombos_level_gap = compute_levels_impact(
    allcombos=allcombos,
    list_impact_cols=list_scenarios_names_impact_breakdown,
    list_shocks=[choice_macro_shock],
    list_responses=choices_macro_response,
    dict_ltavg=dict_ltavg,
)
allcombos_level_gap.to_parquet(
    path_output
    + "allcombos_level_gap_consol_"
    + "shock_response_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".parquet"
)
allcombos_level_gap.to_csv(
    path_output
    + "allcombos_level_gap_consol_"
    + "shock_response_"
    + income_choice
    + "_"
    + outcome_choice
    + "_"
    + fd_suffix
    + hhbasis_suffix
    + ".csv",
    index=False,
)


# VI --- Compile repeated aggregate impact of all combos, but split by response variable
def split_allcombos_heatmap_telegram(
    allcombos,
    impact_cols_total,
    impact_cols_breakdown,
    list_shocks,
    list_responses,
    total_only,
    dict_ltavg,
):
    # Parameters
    if total_only:
        total_only_suffix = "totalonly_"
        list_impact_cols = impact_cols_total
    elif not total_only:
        total_only_suffix = ""
        list_impact_cols = impact_cols_breakdown
    # Deep copies
    allcombos_full = allcombos.copy()
    # Beautify time horizon
    allcombos_full["horizon_year"] = allcombos_full["horizon_year"] + 1
    allcombos_full["horizon_year"] = "Year " + allcombos_full["horizon_year"].astype(
        "str"
    )
    # Split into shock-response specific heatmaps
    list_files = []
    for shock in list_shocks:
        for response in tqdm(list_responses):
            # Extract LT avg
            ltavg = dict_ltavg[response]

            # A. Growth impact
            allcombos_sub = allcombos_full[
                (allcombos_full["shock"] == shock)
                & (allcombos_full["response"] == response)
            ].copy()
            # Keep only time horizon + impact estimates
            for i in ["shock", "response"]:
                del allcombos_sub[i]
            allcombos_sub = allcombos_sub.set_index("horizon_year")
            allcombos_sub = allcombos_sub[list_impact_cols]
            # Generate heatmaps
            fig_allcombos_sub = heatmap(
                input=allcombos_sub,
                mask=False,
                colourmap="vlag",
                outputfile=path_output
                + "allcombos_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
                + ".png",
                title="Breakdown of Impact on " + response + " (pp)",
                lb=0,
                ub=allcombos_sub.max().max(),
                format=".1f",
            )
            list_files = list_files + [
                path_output
                + "allcombos_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
            ]
            # Generate barchart
            allcombos_sub = (
                allcombos_sub.reset_index()
            )  # so that horizon_year is part of the data frame, not index
            bar_allcombos_sub = wide_grouped_barchart(
                data=allcombos_sub,
                y_cols=list_impact_cols,
                group_col="horizon_year",
                main_title="Breakdown of Impact on " + response + " (pp)",
                decimal_points=1,
                group_colours=["lightblue", "lightpink"],
            )
            bar_allcombos_sub.write_image(
                path_output
                + "bar_allcombos_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
                + ".png"
            )
            list_files = list_files + [
                path_output
                + "bar_allcombos_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
            ]

            # B. Levels impact
            # Create data frame to house levels
            allcombos_sub_cf = allcombos_sub.copy()
            allcombos_sub_level = allcombos_sub.copy()
            for horizon in range(0, len(allcombos_sub_cf)):
                allcombos_sub_cf.loc[
                    allcombos_sub_cf["horizon_year"] == "Year " + str(horizon + 1),
                    list_impact_cols,
                ] = 100 * (1 + ltavg / 100) ** (horizon + 1)
            # Compute realised levels
            round_sub_levels = 1
            for horizon in range(0, len(allcombos_sub_cf)):
                if round_sub_levels == 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level["horizon_year"]
                        == "Year " + str(horizon + 1),
                        list_impact_cols,
                    ] = 100 * (
                        1
                        + (
                            ltavg
                            + allcombos_sub.loc[
                                allcombos_sub["horizon_year"]
                                == "Year " + str(horizon + 1),
                                list_impact_cols,
                            ]
                        )
                        / 100
                    )
                elif round_sub_levels > 1:
                    allcombos_sub_level.loc[
                        allcombos_sub_level["horizon_year"]
                        == "Year " + str(horizon + 1),
                        list_impact_cols,
                    ] = allcombos_sub_level[list_impact_cols].shift(1) * (
                        1
                        + (
                            ltavg
                            + allcombos_sub.loc[
                                allcombos_sub["horizon_year"]
                                == "Year " + str(horizon + 1),
                                list_impact_cols,
                            ]
                        )
                        / 100
                    )
                round_sub_levels += 1
            # Compute levels impact
            allcombos_sub_level_gap = allcombos_sub_level.copy()
            allcombos_sub_level_gap[list_impact_cols] = (
                allcombos_sub_level_gap[list_impact_cols]
                - allcombos_sub_cf[list_impact_cols]
            )
            # Create bar chart
            bar_allcombos_sub_level_gap = wide_grouped_barchart(
                data=allcombos_sub_level_gap,
                y_cols=list_impact_cols,
                group_col="horizon_year",
                main_title="Implied Levels Impact (Assuming LT Avg) on "
                + response
                + " (% Baseline)",
                decimal_points=1,
                group_colours=["lightblue", "lightpink"],
            )
            bar_allcombos_sub_level_gap.write_image(
                path_output
                + "bar_allcombos_sub_level_gap_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
                + ".png"
            )
            list_files = list_files + [
                path_output
                + "bar_allcombos_sub_level_gap_"
                + total_only_suffix
                + shock
                + "_"
                + response
                + "_"
                + income_choice
                + "_"
                + outcome_choice
                + "_"
                + fd_suffix
                + hhbasis_suffix
            ]

    # X. Compile PDF
    pil_img2pdf(
        list_images=list_files,
        extension="png",
        pdf_name=path_output
        + "allcombos_"
        + total_only_suffix
        + "shock_response_"
        + income_choice
        + "_"
        + outcome_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix,
    )
    # X. Send telegram
    telsendfiles(
        conf=tel_config,
        path=path_output
        + "allcombos_"
        + total_only_suffix
        + "shock_response_"
        + income_choice
        + "_"
        + outcome_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix
        + ".pdf",
        cap="allcombos_"
        + total_only_suffix
        + "shock_response_"
        + income_choice
        + "_"
        + outcome_choice
        + "_"
        + fd_suffix
        + hhbasis_suffix,
    )


split_allcombos_heatmap_telegram(
    allcombos=allcombos,
    impact_cols_total=list_scenarios_names_total_impact,
    impact_cols_breakdown=list_scenarios_names_impact_breakdown,
    list_shocks=[choice_macro_shock],
    list_responses=choices_macro_response,
    total_only=False,
    dict_ltavg=dict_ltavg,
)

# VI --- Compile repeated aggregate impact of all combos, but split by response variable; total impact only
split_allcombos_heatmap_telegram(
    allcombos=allcombos,
    impact_cols_total=list_scenarios_names_total_impact,
    impact_cols_breakdown=list_scenarios_names_impact_breakdown,
    list_shocks=[choice_macro_shock],
    list_responses=choices_macro_response,
    total_only=True,
    dict_ltavg=dict_ltavg,
)

# X --- Notify
telsendmsg(conf=tel_config, msg="impact-household --- analysis_impact_sim: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
