# %%
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
    boxplot_time,
)
import matplotlib.pyplot as plt
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
path_data = "./data/hies_consol/"
path_2019 = "./data/hies_2022/"
path_output = "./output/"
hhbasis_descriptive = ast.literal_eval(os.getenv("HHBASIS_DESCRIPTIVE"))
equivalised_descriptive = ast.literal_eval(os.getenv("EQUIVALISED_DESCRIPTIVE"))
if hhbasis_descriptive:
    input_suffix = "_hhbasis"
    output_suffix = "_hhbasis"
    chart_suffix = " (Total HH)"
if equivalised_descriptive:
    input_suffix = "_equivalised"
    output_suffix = "_equivalised"
    chart_suffix = " (Equivalised)"
if not hhbasis_descriptive and not equivalised_descriptive:
    input_suffix = ""
    output_suffix = "_capita"
    chart_suffix = " (Per Capita)"

# %%
# I --- Load data
df = pd.read_parquet(
    path_2019 + "hies_2022_consol" + input_suffix + ".parquet"
)  # CHECK: include / exclude outliers and on hhbasis

# %%
# II --- Pre-analysis prep
# Malaysian only


# Generate income groups
def gen_gross_income_group(data, aggregation):
    if aggregation == 1:
        data.loc[
            (data["gross_income"] >= data["gross_income"].quantile(q=0.8)),
            "gross_income_group",
        ] = "4_t20"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.6))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.8))
            ),
            "gross_income_group",
        ] = "3_m20+"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.4))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.6))
            ),
            "gross_income_group",
        ] = "2_m20-"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.2))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.4))
            ),
            "gross_income_group",
        ] = "1_b20+"
        data.loc[
            (data["gross_income"] < data["gross_income"].quantile(q=0.2)),
            "gross_income_group",
        ] = "0_b20-"
    elif aggregation == 2:
        data.loc[
            (data["gross_income"] >= data["gross_income"].quantile(q=0.8)),
            "gross_income_group",
        ] = "2_t20"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.4))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.8))
            ),
            "gross_income_group",
        ] = "1_m40"
        data.loc[
            (data["gross_income"] < data["gross_income"].quantile(q=0.4)),
            "gross_income_group",
        ] = "0_b40"
    elif aggregation == 3:
        data.loc[
            (data["gross_income"] >= data["gross_income"].quantile(q=0.8)),
            "gross_income_group",
        ] = "3_t20"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.6))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.8))
            ),
            "gross_income_group",
        ] = "2_m20+"
        data.loc[
            (
                (data["gross_income"] >= data["gross_income"].quantile(q=0.4))
                & (data["gross_income"] < data["gross_income"].quantile(q=0.6))
            ),
            "gross_income_group",
        ] = "1_m20-"
        data.loc[
            (data["gross_income"] < data["gross_income"].quantile(q=0.4)),
            "gross_income_group",
        ] = "0_b40"


gen_gross_income_group(data=df, aggregation=3)
df["gross_income_group"] = df["gross_income_group"].replace(
    {
        "3_t20": "4_T20",
        "2_m20+": "3_M20+",
        "1_m20-": "2_M20-",
        "1_b40": "0_B40",
    }
)


# %%
# III --- The analysis
# Compute food spending ratio
df["cons_01_inc"] = 100 * df["cons_01"] / df["gross_income"]
# Compile tabulations by HH characteristics
list_wa = [1, 2, 3]
list_child = [0, 1, 2, 3]
list_elderly = [0, 1, 2]
list_file_names = []
for wa in tqdm(list_wa):
    for child in list_child:
        for elderly in list_elderly:
            d = df[
                (df["working_age2"] == wa)
                & (df["kid"] == child)
                & (df["elderly2"] == elderly)
            ]
            # Tabulate median spending by income group
            tab = d.groupby("gross_income_group")[
                ["cons_01", "cons_01_inc"]
            ].median()
            # Generate and send figure
            file_name = (
                path_output
                + "tab_food_spending_"
                + str(wa)
                + "wa"
                + "_"
                + str(child)
                + "child"
                + "_"
                + str(elderly)
                + "elderly"
            )
            list_file_names = list_file_names + [file_name]
            fig = heatmap(
                input=tab,
                mask=False,
                colourmap="vlag",
                outputfile=file_name + ".png",
                title="Median Food Spending by Income Group; "
                + "\n"
                + str(wa)
                + " Working Adults, "
                + str(child)
                + " Child, "
                + str(elderly)
                + " Elderly",
                lb=0,
                ub=tab.max().max(),
                format=".0f",
            )
            # telsendimg(
            #     conf=tel_config,
            #     path=file_name + '.png',
            #     cap=file_name
            # )
# PDF
file_name_pdf = path_output + "tab_food_spending"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
