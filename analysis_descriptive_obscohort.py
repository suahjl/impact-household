# %%
# Produce charts summarising the number of individual-observations per cohort
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
    heatmap_layered,
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
path_data = "data/hies_consol/"
path_output = "./output/"
use_spending_income_ratio = ast.literal_eval(os.getenv("USE_SPENDING_INCOME_RATIO"))

# %% 
# I --- individuals per cohort for main mean and main quantiles
list_bases = ["", "_equivalised", "_hhbasis"]
list_bases_nice = ["Per Capita", "Equivalised", "Total HH"]
list_suffixes = ["mean", "20p", "40p", "60p", "80p", "100p"]
list_suffixes_nice = ["Full Sample", "B20-", "B20+", "M20-", "M20+", "T20"]
for basis, basis_nice in zip(list_bases, list_bases_nice):
    for suffix, suffix_nice in tqdm(zip(list_suffixes, list_suffixes_nice)):
        df = pd.read_parquet(path_data + "hies_consol_agg_obscohort_" + suffix + basis + ".parquet")
        df = df.round(2)
        df = df.set_index("Year")
        df_avg = pd.DataFrame(df["Avg Obs Per Cohort"].copy())
        df_ncohorts = pd.DataFrame(df["Number of Cohorts"].copy())
        # avg
        file_name = path_output + "hies_consol_agg_obscohort_" + suffix + basis + "_avg"
        fig = heatmap(
            input=df_avg,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title="Average Obs Per Cohort (" + basis_nice + "; " + suffix_nice + ")",
            lb=0,
            ub=df_avg.max().max(),
            format=".2f",
            annot_size=36
        )
        # ncohorts
        file_name = path_output + "hies_consol_agg_obscohort_" + suffix + basis + "_ncohorts"
        fig = heatmap(
            input=df_ncohorts,
            mask=False,
            colourmap="vlag",
            outputfile=file_name + ".png",
            title="Number of Cohorts (" + basis_nice + "; " + suffix_nice + ")",
            lb=0,
            ub=df_ncohorts.max().max(),
            format=".0f",
            annot_size=36
        )


# %%
# X --- Notify
# telsendmsg(
#     conf=tel_config,
#     msg="impact-household --- analysis_descriptive_obscohort: COMPLETED",
# )

# %%
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
