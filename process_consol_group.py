# %%
# Merging at the group-level

import pandas as pd
import numpy as np
from helper import telsendmsg, telsendimg, telsendfiles
from tabulate import tabulate
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
path_subgroup = path_data + "subgroup/"
path_2009 = "data/hies_2009/"
path_2014 = "data/hies_2014/"
path_2016 = "data/hies_2016/"
path_2019 = "data/hies_2019/"
# path_2022 = "data/hies_2022/"
path_2022 = "data/hies_2022_bnm/"
hhbasis_cohorts_with_hhsize = ast.literal_eval(os.getenv("HHBASIS_COHORTS_WITH_HHSIZE"))

# %%
# I --- Load processed vintages
df_09 = pd.read_parquet(path_2009 + "hies_2009_consol_trimmedoutliers.parquet")
df_14 = pd.read_parquet(path_2014 + "hies_2014_consol_trimmedoutliers.parquet")
df_16 = pd.read_parquet(path_2016 + "hies_2016_consol_trimmedoutliers.parquet")
df_19 = pd.read_parquet(path_2019 + "hies_2019_consol_trimmedoutliers.parquet")
df_22 = pd.read_parquet(path_2022 + "hies_2022_consol_trimmedoutliers.parquet")

df_14_hhbasis = pd.read_parquet(
    path_2014 + "hies_2014_consol_hhbasis_trimmedoutliers.parquet"
)
df_16_hhbasis = pd.read_parquet(
    path_2016 + "hies_2016_consol_hhbasis_trimmedoutliers.parquet"
)
df_19_hhbasis = pd.read_parquet(
    path_2019 + "hies_2019_consol_hhbasis_trimmedoutliers.parquet"
)
df_22_hhbasis = pd.read_parquet(
    path_2022 + "hies_2022_consol_hhbasis_trimmedoutliers.parquet"
)

df_14_equivalised = pd.read_parquet(
    path_2014 + "hies_2014_consol_equivalised_trimmedoutliers.parquet"
)
df_16_equivalised = pd.read_parquet(
    path_2016 + "hies_2016_consol_equivalised_trimmedoutliers.parquet"
)
df_19_equivalised = pd.read_parquet(
    path_2019 + "hies_2019_consol_equivalised_trimmedoutliers.parquet"
)
df_22_equivalised = pd.read_parquet(
    path_2022 + "hies_2022_consol_equivalised_trimmedoutliers.parquet"
)

df_09_full = pd.read_parquet(path_2009 + "hies_2009_consol.parquet")
df_14_full = pd.read_parquet(path_2014 + "hies_2014_consol.parquet")
df_16_full = pd.read_parquet(path_2016 + "hies_2016_consol.parquet")
df_19_full = pd.read_parquet(path_2019 + "hies_2019_consol.parquet")
df_22_full = pd.read_parquet(path_2022 + "hies_2022_consol.parquet")

df_14_full_hhbasis = pd.read_parquet(path_2014 + "hies_2014_consol_hhbasis.parquet")
df_16_full_hhbasis = pd.read_parquet(path_2016 + "hies_2016_consol_hhbasis.parquet")
df_19_full_hhbasis = pd.read_parquet(path_2019 + "hies_2019_consol_hhbasis.parquet")
df_22_full_hhbasis = pd.read_parquet(path_2022 + "hies_2022_consol_hhbasis.parquet")

df_14_full_equivalised = pd.read_parquet(
    path_2014 + "hies_2014_consol_equivalised.parquet"
)
df_16_full_equivalised = pd.read_parquet(
    path_2016 + "hies_2016_consol_equivalised.parquet"
)
df_19_full_equivalised = pd.read_parquet(
    path_2019 + "hies_2019_consol_equivalised.parquet"
)
df_22_full_equivalised = pd.read_parquet(
    path_2022 + "hies_2022_consol_equivalised.parquet"
)


# %%
# III --- Pre-merger cleaning


# Drop all non-Malaysians (suitable for policy analysis, and that 2022 vintage has no income info for non-Malaysians)
def drop_non_malaysians(data):
    df = data[data["malaysian"] == 1]
    del df["malaysian"]
    return df


df_14 = drop_non_malaysians(data=df_14)
df_16 = drop_non_malaysians(data=df_16)
df_19 = drop_non_malaysians(data=df_19)
df_22 = drop_non_malaysians(data=df_22)

df_14_hhbasis = drop_non_malaysians(data=df_14_hhbasis)
df_16_hhbasis = drop_non_malaysians(data=df_16_hhbasis)
df_19_hhbasis = drop_non_malaysians(data=df_19_hhbasis)
df_22_hhbasis = drop_non_malaysians(data=df_22_hhbasis)

df_14_equivalised = drop_non_malaysians(data=df_14_equivalised)
df_16_equivalised = drop_non_malaysians(data=df_16_equivalised)
df_19_equivalised = drop_non_malaysians(data=df_19_equivalised)
df_22_equivalised = drop_non_malaysians(data=df_22_equivalised)

df_14_full = drop_non_malaysians(data=df_14_full)
df_16_full = drop_non_malaysians(data=df_16_full)
df_19_full = drop_non_malaysians(data=df_19_full)
df_22_full = drop_non_malaysians(data=df_22_full)

df_14_full_hhbasis = drop_non_malaysians(data=df_14_full_hhbasis)
df_16_full_hhbasis = drop_non_malaysians(data=df_16_full_hhbasis)
df_19_full_hhbasis = drop_non_malaysians(data=df_19_full_hhbasis)
df_22_full_hhbasis = drop_non_malaysians(data=df_22_full_hhbasis)

df_14_full_equivalised = drop_non_malaysians(data=df_14_full_equivalised)
df_16_full_equivalised = drop_non_malaysians(data=df_16_full_equivalised)
df_19_full_equivalised = drop_non_malaysians(data=df_19_full_equivalised)
df_22_full_equivalised = drop_non_malaysians(data=df_22_full_equivalised)


# Identify common columns
common_cols_14_22 = list(
    set(df_14.columns) & set(df_16.columns) & set(df_19.columns) & set(df_22.columns)
)
common_col_09_22 = list(
    set(df_09.columns)
    & set(df_14.columns)
    & set(df_16.columns)
    & set(df_19.columns)
    & set(df_22.columns)
)

# Harmonise ethnicity levels (to lowest; 2014)
df_16.loc[~(df_16["ethnicity"] == "bumiputera"), "ethnicity"] = "non_bumiputera"
df_19.loc[~(df_19["ethnicity"] == "bumiputera"), "ethnicity"] = "non_bumiputera"
df_22.loc[~(df_22["ethnicity"] == "bumiputera"), "ethnicity"] = "non_bumiputera"

df_16_hhbasis.loc[
    ~(df_16_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_19_hhbasis.loc[
    ~(df_19_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_22_hhbasis.loc[
    ~(df_22_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"

df_16_equivalised.loc[
    ~(df_16_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_19_equivalised.loc[
    ~(df_19_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_22_equivalised.loc[
    ~(df_22_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"

df_16_full.loc[
    ~(df_16_full["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_19_full.loc[
    ~(df_19_full["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_22_full.loc[
    ~(df_22_full["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"

df_16_full_hhbasis.loc[
    ~(df_16_full_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_19_full_hhbasis.loc[
    ~(df_19_full_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_22_full_hhbasis.loc[
    ~(df_22_full_hhbasis["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"

df_16_full_equivalised.loc[
    ~(df_16_full_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_19_full_equivalised.loc[
    ~(df_19_full_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"
df_22_full_equivalised.loc[
    ~(df_22_full_equivalised["ethnicity"] == "bumiputera"), "ethnicity"
] = "non_bumiputera"

# Not useful
for i in ["id", "svy_weight"]:
    del df_09[i]
    del df_14[i]
    del df_16[i]
    del df_19[i]
    del df_22[i]

    del df_14_hhbasis[i]
    del df_16_hhbasis[i]
    del df_19_hhbasis[i]
    del df_22_hhbasis[i]

    del df_14_equivalised[i]
    del df_16_equivalised[i]
    del df_19_equivalised[i]
    del df_22_equivalised[i]

    del df_09_full[i]
    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]
    del df_22_full[i]

    del df_14_full_hhbasis[i]
    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]
    del df_22_full_hhbasis[i]

    del df_14_full_equivalised[i]
    del df_16_full_equivalised[i]
    del df_19_full_equivalised[i]
    del df_22_full_equivalised[i]


# Buckets for number of income_gen_members, child, and adolescents, and marriage status (age done separately)
# income_gen_members (1, 2, 3+)
def gen_igm_group(data):
    data.loc[data["income_gen_members"] == 1, "income_gen_members_group"] = "1"
    data.loc[data["income_gen_members"] == 2, "income_gen_members_group"] = "2"
    data.loc[data["income_gen_members"] >= 3, "income_gen_members_group"] = "3+"
    # del data['income_gen_members']


gen_igm_group(data=df_14)
gen_igm_group(data=df_16)
gen_igm_group(data=df_19)
gen_igm_group(data=df_22)

gen_igm_group(data=df_14_hhbasis)
gen_igm_group(data=df_16_hhbasis)
gen_igm_group(data=df_19_hhbasis)
gen_igm_group(data=df_22_hhbasis)

gen_igm_group(data=df_14_equivalised)
gen_igm_group(data=df_16_equivalised)
gen_igm_group(data=df_19_equivalised)
gen_igm_group(data=df_22_equivalised)

gen_igm_group(data=df_14_full)
gen_igm_group(data=df_16_full)
gen_igm_group(data=df_19_full)
gen_igm_group(data=df_22_full)

gen_igm_group(data=df_14_full_hhbasis)
gen_igm_group(data=df_16_full_hhbasis)
gen_igm_group(data=df_19_full_hhbasis)
gen_igm_group(data=df_22_full_hhbasis)

gen_igm_group(data=df_14_full_equivalised)
gen_igm_group(data=df_16_full_equivalised)
gen_igm_group(data=df_19_full_equivalised)
gen_igm_group(data=df_22_full_equivalised)


# working age (0, 1, 2, 3+) --- vanilla, not extended definition
def gen_wa_group(data):
    data.loc[data["working_age"] == 0, "working_age_group"] = "0"
    data.loc[data["working_age"] == 1, "working_age_group"] = "1"
    data.loc[data["working_age"] == 2, "working_age_group"] = "2"
    data.loc[data["working_age"] >= 3, "working_age_group"] = "3+"
    # del data['working_age']


gen_wa_group(data=df_14)
gen_wa_group(data=df_16)
gen_wa_group(data=df_19)
gen_wa_group(data=df_22)

gen_wa_group(data=df_14_hhbasis)
gen_wa_group(data=df_16_hhbasis)
gen_wa_group(data=df_19_hhbasis)
gen_wa_group(data=df_22_hhbasis)

gen_wa_group(data=df_14_equivalised)
gen_wa_group(data=df_16_equivalised)
gen_wa_group(data=df_19_equivalised)
gen_wa_group(data=df_22_equivalised)

gen_wa_group(data=df_14_full)
gen_wa_group(data=df_16_full)
gen_wa_group(data=df_19_full)
gen_wa_group(data=df_22_full)

gen_wa_group(data=df_14_full_hhbasis)
gen_wa_group(data=df_16_full_hhbasis)
gen_wa_group(data=df_19_full_hhbasis)
gen_wa_group(data=df_22_full_hhbasis)

gen_wa_group(data=df_14_full_equivalised)
gen_wa_group(data=df_16_full_equivalised)
gen_wa_group(data=df_19_full_equivalised)
gen_wa_group(data=df_22_full_equivalised)


# non-working adult females (0, 1, 2, 3+)
def gen_nwaf_group(data):
    data.loc[
        data["non_working_adult_females"] == 0, "non_working_adult_females_group"
    ] = "0"
    data.loc[
        data["non_working_adult_females"] == 1, "non_working_adult_females_group"
    ] = "1"
    data.loc[
        data["non_working_adult_females"] == 2, "non_working_adult_females_group"
    ] = "2"
    data.loc[
        data["non_working_adult_females"] >= 3, "non_working_adult_females_group"
    ] = "3+"
    # del data['non_working_adult_females']


gen_nwaf_group(data=df_14)
gen_nwaf_group(data=df_16)
gen_nwaf_group(data=df_19)
gen_nwaf_group(data=df_22)

gen_nwaf_group(data=df_14_hhbasis)
gen_nwaf_group(data=df_16_hhbasis)
gen_nwaf_group(data=df_19_hhbasis)
gen_nwaf_group(data=df_22_hhbasis)

gen_nwaf_group(data=df_14_equivalised)
gen_nwaf_group(data=df_16_equivalised)
gen_nwaf_group(data=df_19_equivalised)
gen_nwaf_group(data=df_22_equivalised)

gen_nwaf_group(data=df_14_full)
gen_nwaf_group(data=df_16_full)
gen_nwaf_group(data=df_19_full)
gen_nwaf_group(data=df_22_full)

gen_nwaf_group(data=df_14_full_hhbasis)
gen_nwaf_group(data=df_16_full_hhbasis)
gen_nwaf_group(data=df_19_full_hhbasis)
gen_nwaf_group(data=df_22_full_hhbasis)

gen_nwaf_group(data=df_14_full_equivalised)
gen_nwaf_group(data=df_16_full_equivalised)
gen_nwaf_group(data=df_19_full_equivalised)
gen_nwaf_group(data=df_22_full_equivalised)


# working adult females (0, 1, 2, 3+)
def gen_waf_group(data):
    data.loc[data["working_adult_females"] == 0, "working_adult_females_group"] = "0"
    data.loc[data["working_adult_females"] == 1, "working_adult_females_group"] = "1"
    data.loc[data["working_adult_females"] == 2, "working_adult_females_group"] = "2"
    data.loc[data["working_adult_females"] >= 3, "working_adult_females_group"] = "3+"
    # del data['working_adult_females']


gen_waf_group(data=df_14)
gen_waf_group(data=df_16)
gen_waf_group(data=df_19)
gen_waf_group(data=df_22)

gen_waf_group(data=df_14_hhbasis)
gen_waf_group(data=df_16_hhbasis)
gen_waf_group(data=df_19_hhbasis)
gen_waf_group(data=df_22_hhbasis)

gen_waf_group(data=df_14_equivalised)
gen_waf_group(data=df_16_equivalised)
gen_waf_group(data=df_19_equivalised)
gen_waf_group(data=df_22_equivalised)

gen_waf_group(data=df_14_full)
gen_waf_group(data=df_16_full)
gen_waf_group(data=df_19_full)
gen_waf_group(data=df_22_full)

gen_waf_group(data=df_14_full_hhbasis)
gen_waf_group(data=df_16_full_hhbasis)
gen_waf_group(data=df_19_full_hhbasis)
gen_waf_group(data=df_22_full_hhbasis)

gen_waf_group(data=df_14_full_equivalised)
gen_waf_group(data=df_16_full_equivalised)
gen_waf_group(data=df_19_full_equivalised)
gen_waf_group(data=df_22_full_equivalised)


# child (1, 2, 3+)
def gen_child_group(data):
    data.loc[data["child"] == 0, "child_group"] = "0"
    data.loc[data["child"] == 1, "child_group"] = "1"
    data.loc[data["child"] == 2, "child_group"] = "2"
    data.loc[data["child"] >= 3, "child_group"] = "3+"
    # del data['child']


gen_child_group(data=df_14)
gen_child_group(data=df_16)
gen_child_group(data=df_19)
gen_child_group(data=df_22)

gen_child_group(data=df_14_hhbasis)
gen_child_group(data=df_16_hhbasis)
gen_child_group(data=df_19_hhbasis)
gen_child_group(data=df_22_hhbasis)

gen_child_group(data=df_14_equivalised)
gen_child_group(data=df_16_equivalised)
gen_child_group(data=df_19_equivalised)
gen_child_group(data=df_22_equivalised)

gen_child_group(data=df_14_full)
gen_child_group(data=df_16_full)
gen_child_group(data=df_19_full)
gen_child_group(data=df_22_full)

gen_child_group(data=df_14_full_hhbasis)
gen_child_group(data=df_16_full_hhbasis)
gen_child_group(data=df_19_full_hhbasis)
gen_child_group(data=df_22_full_hhbasis)

gen_child_group(data=df_14_full_equivalised)
gen_child_group(data=df_16_full_equivalised)
gen_child_group(data=df_19_full_equivalised)
gen_child_group(data=df_22_full_equivalised)


# adolescents (1, 2, 3+)
def gen_adolescent_group(data):
    data.loc[data["adolescent"] == 0, "adolescent_group"] = "0"
    data.loc[data["adolescent"] == 1, "adolescent_group"] = "1"
    data.loc[data["adolescent"] == 2, "adolescent_group"] = "2"
    data.loc[data["adolescent"] >= 3, "adolescent_group"] = "3+"
    # del data['adolescent']


gen_adolescent_group(data=df_14)
gen_adolescent_group(data=df_16)
gen_adolescent_group(data=df_19)
gen_adolescent_group(data=df_22)

gen_adolescent_group(data=df_14_hhbasis)
gen_adolescent_group(data=df_16_hhbasis)
gen_adolescent_group(data=df_19_hhbasis)
gen_adolescent_group(data=df_22_hhbasis)

gen_adolescent_group(data=df_14_equivalised)
gen_adolescent_group(data=df_16_equivalised)
gen_adolescent_group(data=df_19_equivalised)
gen_adolescent_group(data=df_22_equivalised)

gen_adolescent_group(data=df_14_full)
gen_adolescent_group(data=df_16_full)
gen_adolescent_group(data=df_19_full)
gen_adolescent_group(data=df_22_full)

gen_adolescent_group(data=df_14_full_hhbasis)
gen_adolescent_group(data=df_16_full_hhbasis)
gen_adolescent_group(data=df_19_full_hhbasis)
gen_adolescent_group(data=df_22_full_hhbasis)

gen_adolescent_group(data=df_14_full_equivalised)
gen_adolescent_group(data=df_16_full_equivalised)
gen_adolescent_group(data=df_19_full_equivalised)
gen_adolescent_group(data=df_22_full_equivalised)


# kids (1, 2, 3, 4, 6+)
def gen_kid_group(data):
    data.loc[data["kid"] == 0, "kid_group"] = "0"
    data.loc[data["kid"] == 1, "kid_group"] = "1"
    data.loc[data["kid"] == 2, "kid_group"] = "2"
    data.loc[data["kid"] >= 3, "kid_group"] = "3+"
    # del data['kid']


gen_kid_group(data=df_14)
gen_kid_group(data=df_16)
gen_kid_group(data=df_19)
gen_kid_group(data=df_22)

gen_kid_group(data=df_14_hhbasis)
gen_kid_group(data=df_16_hhbasis)
gen_kid_group(data=df_19_hhbasis)
gen_kid_group(data=df_22_hhbasis)

gen_kid_group(data=df_14_equivalised)
gen_kid_group(data=df_16_equivalised)
gen_kid_group(data=df_19_equivalised)
gen_kid_group(data=df_22_equivalised)

gen_kid_group(data=df_14_full)
gen_kid_group(data=df_16_full)
gen_kid_group(data=df_19_full)
gen_kid_group(data=df_22_full)

gen_kid_group(data=df_14_full_hhbasis)
gen_kid_group(data=df_16_full_hhbasis)
gen_kid_group(data=df_19_full_hhbasis)
gen_kid_group(data=df_22_full_hhbasis)

gen_kid_group(data=df_14_full_equivalised)
gen_kid_group(data=df_16_full_equivalised)
gen_kid_group(data=df_19_full_equivalised)
gen_kid_group(data=df_22_full_equivalised)


# elderly group (1, 2, 3+)
def gen_elderly_group(data):
    data.loc[data["elderly"] == 0, "elderly_group"] = "0"
    data.loc[data["elderly"] == 1, "elderly_group"] = "1"
    data.loc[data["elderly"] == 2, "elderly_group"] = "2"
    data.loc[data["elderly"] >= 3, "elderly_group"] = "3+"
    # del data['elderly']


gen_elderly_group(data=df_14)
gen_elderly_group(data=df_16)
gen_elderly_group(data=df_19)
gen_elderly_group(data=df_22)

gen_elderly_group(data=df_14_hhbasis)
gen_elderly_group(data=df_16_hhbasis)
gen_elderly_group(data=df_19_hhbasis)
gen_elderly_group(data=df_22_hhbasis)

gen_elderly_group(data=df_14_equivalised)
gen_elderly_group(data=df_16_equivalised)
gen_elderly_group(data=df_19_equivalised)
gen_elderly_group(data=df_22_equivalised)

gen_elderly_group(data=df_14_full)
gen_elderly_group(data=df_16_full)
gen_elderly_group(data=df_19_full)
gen_elderly_group(data=df_22_full)

gen_elderly_group(data=df_14_full_hhbasis)
gen_elderly_group(data=df_16_full_hhbasis)
gen_elderly_group(data=df_19_full_hhbasis)
gen_elderly_group(data=df_22_full_hhbasis)

gen_elderly_group(data=df_14_full_equivalised)
gen_elderly_group(data=df_16_full_equivalised)
gen_elderly_group(data=df_19_full_equivalised)
gen_elderly_group(data=df_22_full_equivalised)


# collapse marriage groups
def collapse_marriage(data):
    data.loc[
        (
            (data["marriage"] == "separated")
            | (data["marriage"] == "divorced")
            | (data["marriage"] == "never")
            | (data["marriage"] == "widowed")
        ),
        "marriage",
    ] = "single"


collapse_marriage(data=df_14)
collapse_marriage(data=df_16)
collapse_marriage(data=df_19)
collapse_marriage(data=df_22)

collapse_marriage(data=df_14_hhbasis)
collapse_marriage(data=df_16_hhbasis)
collapse_marriage(data=df_19_hhbasis)
collapse_marriage(data=df_22_hhbasis)

collapse_marriage(data=df_14_equivalised)
collapse_marriage(data=df_16_equivalised)
collapse_marriage(data=df_19_equivalised)
collapse_marriage(data=df_22_equivalised)

collapse_marriage(data=df_14_full)
collapse_marriage(data=df_16_full)
collapse_marriage(data=df_19_full)
collapse_marriage(data=df_22_full)

collapse_marriage(data=df_14_full_hhbasis)
collapse_marriage(data=df_16_full_hhbasis)
collapse_marriage(data=df_19_full_hhbasis)
collapse_marriage(data=df_22_full_hhbasis)

collapse_marriage(data=df_14_full_equivalised)
collapse_marriage(data=df_16_full_equivalised)
collapse_marriage(data=df_19_full_equivalised)
collapse_marriage(data=df_22_full_equivalised)


# collapse education
def collapse_education(data):
    data.loc[
        (
            (data["education"] == "stpm")
            | (data["education"] == "spm")
            | (data["education"] == "pmr")
        ),
        "education",
    ] = "school"


collapse_education(data=df_14)
collapse_education(data=df_16)
collapse_education(data=df_19)
collapse_education(data=df_22)

collapse_education(data=df_14_hhbasis)
collapse_education(data=df_16_hhbasis)
collapse_education(data=df_19_hhbasis)
collapse_education(data=df_22_hhbasis)

collapse_education(data=df_14_equivalised)
collapse_education(data=df_16_equivalised)
collapse_education(data=df_19_equivalised)
collapse_education(data=df_22_equivalised)

collapse_education(data=df_14_full)
collapse_education(data=df_16_full)
collapse_education(data=df_19_full)
collapse_education(data=df_22_full)

collapse_education(data=df_14_full_hhbasis)
collapse_education(data=df_16_full_hhbasis)
collapse_education(data=df_19_full_hhbasis)
collapse_education(data=df_22_full_hhbasis)

collapse_education(data=df_14_full_equivalised)
collapse_education(data=df_16_full_equivalised)
collapse_education(data=df_19_full_equivalised)
collapse_education(data=df_22_full_equivalised)


# collapse emp_status
def collapse_emp_status(data):
    data.loc[
        (
            (data["emp_status"] == "housespouse")
            | (data["emp_status"] == "unemployed")
            | (data["emp_status"] == "unpaid_fam")
            | (data["emp_status"] == "child_not_at_work")
        ),
        "emp_status",
    ] = "no_paid_work"


collapse_emp_status(data=df_14)
collapse_emp_status(data=df_16)
collapse_emp_status(data=df_19)
collapse_emp_status(data=df_22)

collapse_emp_status(data=df_14_hhbasis)
collapse_emp_status(data=df_16_hhbasis)
collapse_emp_status(data=df_19_hhbasis)
collapse_emp_status(data=df_22_hhbasis)

collapse_emp_status(data=df_14_equivalised)
collapse_emp_status(data=df_16_equivalised)
collapse_emp_status(data=df_19_equivalised)
collapse_emp_status(data=df_22_equivalised)

collapse_emp_status(data=df_14_full)
collapse_emp_status(data=df_16_full)
collapse_emp_status(data=df_19_full)
collapse_emp_status(data=df_22_full)

collapse_emp_status(data=df_14_full_hhbasis)
collapse_emp_status(data=df_16_full_hhbasis)
collapse_emp_status(data=df_19_full_hhbasis)
collapse_emp_status(data=df_22_full_hhbasis)

collapse_emp_status(data=df_14_full_equivalised)
collapse_emp_status(data=df_16_full_equivalised)
collapse_emp_status(data=df_19_full_equivalised)
collapse_emp_status(data=df_22_full_equivalised)


# age groups
def gen_age_group(data, aggregation):
    if aggregation == 1:
        data.loc[(data["age"] <= 29), "age_group"] = "0_29"
        data.loc[((data["age"] >= 30) & (data["age"] <= 39)), "age_group"] = "30_39"
        data.loc[((data["age"] >= 40) & (data["age"] <= 49)), "age_group"] = "40_49"
        data.loc[((data["age"] >= 50) & (data["age"] <= 59)), "age_group"] = "50_59"
        data.loc[((data["age"] >= 60) & (data["age"] <= 69)), "age_group"] = "60_69"
        data.loc[(data["age"] >= 70), "age_group"] = "70+"
    elif aggregation == 2:
        data.loc[(data["age"] <= 39), "age_group"] = "0_39"
        data.loc[((data["age"] >= 40) & (data["age"] <= 59)), "age_group"] = "40_59"
        data.loc[(data["age"] >= 60), "age_group"] = "60+"
    # del data['age']


gen_age_group(data=df_14, aggregation=2)
gen_age_group(data=df_16, aggregation=2)
gen_age_group(data=df_19, aggregation=2)
gen_age_group(data=df_22, aggregation=2)

gen_age_group(data=df_14_hhbasis, aggregation=2)
gen_age_group(data=df_16_hhbasis, aggregation=2)
gen_age_group(data=df_19_hhbasis, aggregation=2)
gen_age_group(data=df_22_hhbasis, aggregation=2)

gen_age_group(data=df_14_equivalised, aggregation=2)
gen_age_group(data=df_16_equivalised, aggregation=2)
gen_age_group(data=df_19_equivalised, aggregation=2)
gen_age_group(data=df_22_equivalised, aggregation=2)

gen_age_group(data=df_14_full, aggregation=2)
gen_age_group(data=df_16_full, aggregation=2)
gen_age_group(data=df_19_full, aggregation=2)
gen_age_group(data=df_22_full, aggregation=2)

gen_age_group(data=df_14_full_hhbasis, aggregation=2)
gen_age_group(data=df_16_full_hhbasis, aggregation=2)
gen_age_group(data=df_19_full_hhbasis, aggregation=2)
gen_age_group(data=df_22_full_hhbasis, aggregation=2)

gen_age_group(data=df_14_full_equivalised, aggregation=2)
gen_age_group(data=df_16_full_equivalised, aggregation=2)
gen_age_group(data=df_19_full_equivalised, aggregation=2)
gen_age_group(data=df_22_full_equivalised, aggregation=2)


# Birth year groups
def gen_birth_year_group(data, aggregation):
    if aggregation == 1:
        data.loc[(data["birth_year"] >= 1990), "birth_year_group"] = "1990+"
        data.loc[
            ((data["birth_year"] >= 1980) & (data["birth_year"] <= 1989)),
            "birth_year_group",
        ] = "1980s"
        data.loc[
            ((data["birth_year"] >= 1970) & (data["birth_year"] <= 1979)),
            "birth_year_group",
        ] = "1970s"
        data.loc[
            ((data["birth_year"] >= 1960) & (data["birth_year"] <= 1969)),
            "birth_year_group",
        ] = "1960s"
        data.loc[
            ((data["birth_year"] >= 1950) & (data["birth_year"] <= 1959)),
            "birth_year_group",
        ] = "1950s"
        data.loc[(data["birth_year"] <= 1949), "birth_year_group"] = "1949-"
    elif aggregation == 2:
        data.loc[(data["birth_year"] >= 1980), "birth_year_group"] = "1980+"
        data.loc[
            ((data["birth_year"] >= 1960) & (data["birth_year"] <= 1979)),
            "birth_year_group",
        ] = "1960_79"
        data.loc[(data["birth_year"] <= 1959), "birth_year_group"] = "1959-"
    # del data['birth_year']


gen_birth_year_group(data=df_14, aggregation=2)
gen_birth_year_group(data=df_16, aggregation=2)
gen_birth_year_group(data=df_19, aggregation=2)
gen_birth_year_group(data=df_22, aggregation=2)

gen_birth_year_group(data=df_14_hhbasis, aggregation=2)
gen_birth_year_group(data=df_16_hhbasis, aggregation=2)
gen_birth_year_group(data=df_19_hhbasis, aggregation=2)
gen_birth_year_group(data=df_22_hhbasis, aggregation=2)

gen_birth_year_group(data=df_14_equivalised, aggregation=2)
gen_birth_year_group(data=df_16_equivalised, aggregation=2)
gen_birth_year_group(data=df_19_equivalised, aggregation=2)
gen_birth_year_group(data=df_22_equivalised, aggregation=2)

gen_birth_year_group(data=df_14_full, aggregation=2)
gen_birth_year_group(data=df_16_full, aggregation=2)
gen_birth_year_group(data=df_19_full, aggregation=2)
gen_birth_year_group(data=df_22_full, aggregation=2)

gen_birth_year_group(data=df_14_full_hhbasis, aggregation=2)
gen_birth_year_group(data=df_16_full_hhbasis, aggregation=2)
gen_birth_year_group(data=df_19_full_hhbasis, aggregation=2)
gen_birth_year_group(data=df_22_full_hhbasis, aggregation=2)

gen_birth_year_group(data=df_14_full_equivalised, aggregation=2)
gen_birth_year_group(data=df_16_full_equivalised, aggregation=2)
gen_birth_year_group(data=df_19_full_equivalised, aggregation=2)
gen_birth_year_group(data=df_22_full_equivalised, aggregation=2)


# Household size groups (only for household basis)
def gen_hh_size_group(data):
    data["hh_size_group"] = data["hh_size"].copy()
    data.loc[data["hh_size"] >= 8, "hh_size_group"] = "8+"
    data["hh_size_group"] = data["hh_size_group"].astype("str")


gen_hh_size_group(data=df_14_hhbasis)
gen_hh_size_group(data=df_16_hhbasis)
gen_hh_size_group(data=df_19_hhbasis)
gen_hh_size_group(data=df_22_hhbasis)

gen_hh_size_group(data=df_14_full_hhbasis)
gen_hh_size_group(data=df_16_full_hhbasis)
gen_hh_size_group(data=df_19_full_hhbasis)
gen_hh_size_group(data=df_22_full_hhbasis)


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


gen_gross_income_group(data=df_14, aggregation=1)
gen_gross_income_group(data=df_16, aggregation=1)
gen_gross_income_group(data=df_19, aggregation=1)
gen_gross_income_group(data=df_22, aggregation=1)

gen_gross_income_group(data=df_14_hhbasis, aggregation=1)
gen_gross_income_group(data=df_16_hhbasis, aggregation=1)
gen_gross_income_group(data=df_19_hhbasis, aggregation=1)
gen_gross_income_group(data=df_22_hhbasis, aggregation=1)

gen_gross_income_group(data=df_14_equivalised, aggregation=1)
gen_gross_income_group(data=df_16_equivalised, aggregation=1)
gen_gross_income_group(data=df_19_equivalised, aggregation=1)
gen_gross_income_group(data=df_22_equivalised, aggregation=1)

gen_gross_income_group(data=df_14_full, aggregation=1)
gen_gross_income_group(data=df_16_full, aggregation=1)
gen_gross_income_group(data=df_19_full, aggregation=1)
gen_gross_income_group(data=df_22_full, aggregation=1)

gen_gross_income_group(data=df_14_full_hhbasis, aggregation=1)
gen_gross_income_group(data=df_16_full_hhbasis, aggregation=1)
gen_gross_income_group(data=df_19_full_hhbasis, aggregation=1)
gen_gross_income_group(data=df_22_full_hhbasis, aggregation=1)

gen_gross_income_group(data=df_14_full_equivalised, aggregation=1)
gen_gross_income_group(data=df_16_full_equivalised, aggregation=1)
gen_gross_income_group(data=df_19_full_equivalised, aggregation=1)
gen_gross_income_group(data=df_22_full_equivalised, aggregation=1)

# Delete redundant columns; keep hh_size for hhbasis dataset
for i in ["hh_size"]:
    del df_14[i]
    del df_16[i]
    del df_19[i]
    del df_22[i]

    del df_14_equivalised[i]
    del df_16_equivalised[i]
    del df_19_equivalised[i]
    del df_22_equivalised[i]

    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]
    del df_22_full[i]

    del df_14_full_equivalised[i]
    del df_16_full_equivalised[i]
    del df_19_full_equivalised[i]
    del df_22_full_equivalised[i]

for i in [
    "monthly_income",
    "net_income",
    "net_transfers",
    "net_margin",
]:  # already removed for 2022 vintage
    del df_16[i]
    del df_19[i]

    del df_16_hhbasis[i]
    del df_19_hhbasis[i]

    del df_16_equivalised[i]
    del df_19_equivalised[i]

    del df_16_full[i]
    del df_19_full[i]

    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]

    del df_16_full_equivalised[i]
    del df_19_full_equivalised[i]

# Create copy of dataframes with base columns that have been transformed into categoricals
df_14_withbase = df_14.copy()
df_16_withbase = df_16.copy()
df_19_withbase = df_19.copy()
df_22_withbase = df_22.copy()

df_14_hhbasis_withbase = df_14_hhbasis.copy()
df_16_hhbasis_withbase = df_16_hhbasis.copy()
df_19_hhbasis_withbase = df_19_hhbasis.copy()
df_22_hhbasis_withbase = df_22_hhbasis.copy()

df_14_equivalised_withbase = df_14_equivalised.copy()
df_16_equivalised_withbase = df_16_equivalised.copy()
df_19_equivalised_withbase = df_19_equivalised.copy()
df_22_equivalised_withbase = df_22_equivalised.copy()

df_14_full_withbase = df_14_full.copy()
df_16_full_withbase = df_16_full.copy()
df_19_full_withbase = df_19_full.copy()
df_22_full_withbase = df_22_full.copy()

df_14_full_hhbasis_withbase = df_14_full_hhbasis.copy()
df_16_full_hhbasis_withbase = df_16_full_hhbasis.copy()
df_19_full_hhbasis_withbase = df_19_full_hhbasis.copy()
df_22_full_hhbasis_withbase = df_22_full_hhbasis.copy()

df_14_full_equivalised_withbase = df_14_full_equivalised.copy()
df_16_full_equivalised_withbase = df_16_full_equivalised.copy()
df_19_full_equivalised_withbase = df_19_full_equivalised.copy()
df_22_full_equivalised_withbase = df_22_full_equivalised.copy()

# Save copy of dataframes with base columns that have been transformed into categoricals
df_14_withbase.to_parquet(
    path_2014 + "hies_2014_consol_trimmedoutliers_groupandbase.parquet"
)
df_16_withbase.to_parquet(
    path_2016 + "hies_2016_consol_trimmedoutliers_groupandbase.parquet"
)
df_19_withbase.to_parquet(
    path_2019 + "hies_2019_consol_trimmedoutliers_groupandbase.parquet"
)
df_22_withbase.to_parquet(
    path_2022 + "hies_2022_consol_trimmedoutliers_groupandbase.parquet"
)

df_14_hhbasis_withbase.to_parquet(
    path_2014 + "hies_2014_consol_hhbasis_trimmedoutliers_groupandbase.parquet"
)
df_16_hhbasis_withbase.to_parquet(
    path_2016 + "hies_2016_consol_hhbasis_trimmedoutliers_groupandbase.parquet"
)
df_19_hhbasis_withbase.to_parquet(
    path_2019 + "hies_2019_consol_hhbasis_trimmedoutliers_groupandbase.parquet"
)
df_22_hhbasis_withbase.to_parquet(
    path_2022 + "hies_2022_consol_hhbasis_trimmedoutliers_groupandbase.parquet"
)

df_14_equivalised_withbase.to_parquet(
    path_2014 + "hies_2014_consol_equivalised_trimmedoutliers_groupandbase.parquet"
)
df_16_equivalised_withbase.to_parquet(
    path_2016 + "hies_2016_consol_equivalised_trimmedoutliers_groupandbase.parquet"
)
df_19_equivalised_withbase.to_parquet(
    path_2019 + "hies_2019_consol_equivalised_trimmedoutliers_groupandbase.parquet"
)
df_22_equivalised_withbase.to_parquet(
    path_2022 + "hies_2022_consol_equivalised_trimmedoutliers_groupandbase.parquet"
)

df_14_full_withbase.to_parquet(path_2014 + "hies_2014_consol_groupandbase.parquet")
df_16_full_withbase.to_parquet(path_2016 + "hies_2016_consol_groupandbase.parquet")
df_19_full_withbase.to_parquet(path_2019 + "hies_2019_consol_groupandbase.parquet")
df_22_full_withbase.to_parquet(path_2022 + "hies_2022_consol_groupandbase.parquet")

df_14_full_hhbasis_withbase.to_parquet(
    path_2014 + "hies_2014_consol_hhbasis_groupandbase.parquet"
)
df_16_full_hhbasis_withbase.to_parquet(
    path_2016 + "hies_2016_consol_hhbasis_groupandbase.parquet"
)
df_19_full_hhbasis_withbase.to_parquet(
    path_2019 + "hies_2019_consol_hhbasis_groupandbase.parquet"
)
df_22_full_hhbasis_withbase.to_parquet(
    path_2022 + "hies_2022_consol_hhbasis_groupandbase.parquet"
)

df_14_full_equivalised_withbase.to_parquet(
    path_2014 + "hies_2014_consol_equivalised_groupandbase.parquet"
)
df_16_full_equivalised_withbase.to_parquet(
    path_2016 + "hies_2016_consol_equivalised_groupandbase.parquet"
)
df_19_full_equivalised_withbase.to_parquet(
    path_2019 + "hies_2019_consol_equivalised_groupandbase.parquet"
)
df_22_full_equivalised_withbase.to_parquet(
    path_2022 + "hies_2022_consol_equivalised_groupandbase.parquet"
)

# Delete base columns that have been transformed into categoricals
cols_base_group_transformed = [
    "age",
    "income_gen_members",
    "non_working_adult_females",
    "working_adult_females",
    "working_age",
    "kid",
    "child",
    "adolescent",
    "elderly",
    "birth_year",
]
if hhbasis_cohorts_with_hhsize:
    cols_base_group_transformed_with_hhsize = cols_base_group_transformed + ["hh_size"]
elif not hhbasis_cohorts_with_hhsize:
    cols_base_group_transformed_with_hhsize = cols_base_group_transformed.copy()
for i in cols_base_group_transformed:
    del df_14[i]
    del df_16[i]
    del df_19[i]
    del df_22[i]

    del df_14_full[i]
    del df_16_full[i]
    del df_19_full[i]
    del df_22_full[i]
for i in cols_base_group_transformed_with_hhsize:
    del df_14_hhbasis[i]
    del df_16_hhbasis[i]
    del df_19_hhbasis[i]
    del df_22_hhbasis[i]

    del df_14_equivalised[i]
    del df_16_equivalised[i]
    del df_19_equivalised[i]
    del df_22_equivalised[i]

    del df_14_full_hhbasis[i]
    del df_16_full_hhbasis[i]
    del df_19_full_hhbasis[i]
    del df_22_full_hhbasis[i]

    del df_14_full_equivalised[i]
    del df_16_full_equivalised[i]
    del df_19_full_equivalised[i]
    del df_22_full_equivalised[i]

# %%
# IV.A --- Merger (group-level; 2014 - 2019)
# Outcome variables to be collapsed
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
    "cons_0451_elec",
    "cons_07_ex_bigticket",
]
col_inc = [
    "salaried_wages",
    "other_wages",
    "asset_income",
    "gross_transfers",
    "gross_income",
]
col_outcomes = col_cons + col_inc
# Observables to be grouped on (exclude some overlapping variables like kid, working_age2, elderly2)
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
if hhbasis_cohorts_with_hhsize:  # total household basis
    col_groups_with_hhsize = col_groups + ["hh_size_group"]
elif not hhbasis_cohorts_with_hhsize:  # whether per capita or equivalised
    col_groups_with_hhsize = col_groups.copy()


# Functions
def gen_pseudopanel(
    data1,
    data2,
    data3,
    data4,
    list_cols_cohort,
    list_cols_outcomes,
    use_mean,
    use_quantile,
    quantile_choice,
    file_suffix,
    base_path,
):
    # Groupby operation
    if use_mean and not use_quantile:
        # generate pseudo panel time periods
        df1_agg = (
            data1.groupby(list_cols_cohort)[list_cols_outcomes]
            .mean(numeric_only=True)
            .reset_index()
        )
        df2_agg = (
            data2.groupby(list_cols_cohort)[list_cols_outcomes]
            .mean(numeric_only=True)
            .reset_index()
        )
        df3_agg = (
            data3.groupby(list_cols_cohort)[list_cols_outcomes]
            .mean(numeric_only=True)
            .reset_index()
        )
        df4_agg = (
            data4.groupby(list_cols_cohort)[list_cols_outcomes]
            .mean(numeric_only=True)
            .reset_index()
        )
    elif use_quantile and not use_mean:
        # generate pseudo panel time periods
        df1_agg = (
            data1.groupby(list_cols_cohort)[list_cols_outcomes]
            .quantile(numeric_only=True, q=quantile_choice)
            .reset_index()
        )
        df2_agg = (
            data2.groupby(list_cols_cohort)[list_cols_outcomes]
            .quantile(numeric_only=True, q=quantile_choice)
            .reset_index()
        )
        df3_agg = (
            data3.groupby(list_cols_cohort)[list_cols_outcomes]
            .quantile(numeric_only=True, q=quantile_choice)
            .reset_index()
        )
        df4_agg = (
            data4.groupby(list_cols_cohort)[list_cols_outcomes]
            .quantile(numeric_only=True, q=quantile_choice)
            .reset_index()
        )
    elif use_mean and use_quantile:
        raise NotImplementedError("Use either mean or quantiles")
    elif not use_mean and not use_quantile:
        raise NotImplementedError("Use either mean or quantiles")
    # Generate time identifiers
    df1_agg["_time"] = 1
    df2_agg["_time"] = 2
    df3_agg["_time"] = 3
    df4_agg["_time"] = 4
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg, df4_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ["_time"]).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(
        df2_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df3_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df4_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced["balanced"] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how="left")
    df_agg_balanced = df_agg_balanced[df_agg_balanced["balanced"] == 1]
    del df_agg_balanced["balanced"]
    df_agg_balanced = df_agg_balanced.sort_values(
        by=list_cols_cohort + ["_time"]
    ).reset_index(drop=True)
    
    # calculate number of individual observations per cohort
    obscohort_df1 = data1.groupby(list_cols_cohort).size()
    obscohort_df2 = data2.groupby(list_cols_cohort).size()
    obscohort_df3 = data1.groupby(list_cols_cohort).size()
    obscohort_df4 = data1.groupby(list_cols_cohort).size()
    obscohort_all = pd.concat(
        [obscohort_df1, obscohort_df2, obscohort_df3, obscohort_df4], axis=0
    )
    obscohort = pd.DataFrame(
        {
            "Year": ["2014", "2016", "2019", "2022", "Total"],
            "Avg Obs Per Cohort": [
                len(data1) / len(df1_agg),
                len(data2) / len(df2_agg),
                len(data3) / len(df3_agg),
                len(data4) / len(df4_agg),
                (len(data1) + len(data2) + len(data3) + len(data4)) / len(df_agg_balanced) / 4,
            ],
            "Number of Cohorts": [
                len(df1_agg),
                len(df2_agg),
                len(df3_agg),
                len(df4_agg),
                len(df_agg_balanced) / 4,
            ],
        }
    )
    # Save to local
    df_agg.to_parquet(base_path + "hies_consol_agg_" + file_suffix + ".parquet")
    df_agg_balanced.to_parquet(
        base_path + "hies_consol_agg_balanced_" + file_suffix + ".parquet"
    )
    obscohort.to_parquet(
        base_path + "hies_consol_agg_obscohort_" + file_suffix + ".parquet"
    )
    # Output
    return df_agg, df_agg_balanced


def gen_pseudopanel_quantile_fixed_axis(
    data1,
    data2,
    data3,
    data4,
    list_cols_cohort,
    list_cols_outcomes,
    fixed_axis,
    quantile_choice,
    file_suffix,
    base_path,
):
    # Prelims
    df1 = data1.copy()
    df2 = data2.copy()
    df3 = data3.copy()
    df4 = data4.copy()
    # Create reference point on fixed axis
    df1_ref = (
        data1.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=quantile_choice, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_fixed_axis"})
    )
    df2_ref = (
        data2.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=quantile_choice, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_fixed_axis"})
    )
    df3_ref = (
        data3.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=quantile_choice, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_fixed_axis"})
    )
    df4_ref = (
        data4.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=quantile_choice, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_fixed_axis"})
    )
    # Merge back
    df1 = df1.merge(df1_ref, on=list_cols_cohort, how="left")
    df2 = df2.merge(df2_ref, on=list_cols_cohort, how="left")
    df3 = df3.merge(df3_ref, on=list_cols_cohort, how="left")
    df4 = df4.merge(df4_ref, on=list_cols_cohort, how="left")
    # Keep rows where quantile ref point match
    df1 = df1[df1[fixed_axis] == df1["_fixed_axis"]]
    df2 = df2[df2[fixed_axis] == df2["_fixed_axis"]]
    df3 = df3[df3[fixed_axis] == df3["_fixed_axis"]]
    df4 = df4[df4[fixed_axis] == df4["_fixed_axis"]]
    # Collapse
    df1_agg = (
        df1.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df2_agg = (
        df2.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df3_agg = (
        df3.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df4_agg = (
        df4.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    # Generate time identifiers
    df1_agg["_time"] = 1
    df2_agg["_time"] = 2
    df3_agg["_time"] = 3
    df4_agg["_time"] = 4
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg, df4_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ["_time"]).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(
        df2_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df3_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df4_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced["balanced"] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how="left")
    df_agg_balanced = df_agg_balanced[df_agg_balanced["balanced"] == 1]
    del df_agg_balanced["balanced"]
    df_agg_balanced = df_agg_balanced.sort_values(
        by=list_cols_cohort + ["_time"]
    ).reset_index(drop=True)
    
    # calculate number of individual observations per cohort
    obscohort_df1 = data1.groupby(list_cols_cohort).size()
    obscohort_df2 = data2.groupby(list_cols_cohort).size()
    obscohort_df3 = data1.groupby(list_cols_cohort).size()
    obscohort_df4 = data1.groupby(list_cols_cohort).size()
    obscohort_all = pd.concat(
        [obscohort_df1, obscohort_df2, obscohort_df3, obscohort_df4], axis=0
    )
    obscohort = pd.DataFrame(
        {
            "Year": ["2014", "2016", "2019", "2022", "Total"],
            "Avg Obs Per Cohort": [
                len(data1) / len(df1_agg),
                len(data2) / len(df2_agg),
                len(data3) / len(df3_agg),
                len(data4) / len(df4_agg),
                (len(data1) + len(data2) + len(data3) + len(data4)) / len(df_agg_balanced) / 4,
            ],
            "Number of Cohorts": [
                len(df1_agg),
                len(df2_agg),
                len(df3_agg),
                len(df4_agg),
                len(df_agg_balanced) / 4,
            ],
        }
    )
    # Save to local
    df_agg.to_parquet(base_path + "hies_consol_agg_" + file_suffix + ".parquet")
    df_agg_balanced.to_parquet(
        base_path + "hies_consol_agg_balanced_" + file_suffix + ".parquet"
    )
    obscohort.to_parquet(
        base_path + "hies_consol_agg_obscohort_" + file_suffix + ".parquet"
    )
    # Output
    return df_agg, df_agg_balanced


def gen_pseudopanel_distgroup_fixed_axis(
    data1,
    data2,
    data3,
    data4,
    list_cols_cohort,
    list_cols_outcomes,
    fixed_axis,
    distbounds,
    file_suffix,
    base_path,
):
    # Prelims
    df1 = data1.copy()
    df2 = data2.copy()
    df3 = data3.copy()
    df4 = data4.copy()

    q_lb = distbounds[0]  # lower bound of distribution
    q_ub = distbounds[1]  # upper bound of distribution
    # Create reference points (LB and UB) on fixed axis
    df1_lb = (
        data1.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_lb, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_lb_fixed_axis"})
    )
    df1_ub = (
        data1.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_ub, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_ub_fixed_axis"})
    )

    df2_lb = (
        data2.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_lb, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_lb_fixed_axis"})
    )
    df2_ub = (
        data2.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_ub, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_ub_fixed_axis"})
    )

    df3_lb = (
        data3.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_lb, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_lb_fixed_axis"})
    )
    df3_ub = (
        data3.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_ub, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_ub_fixed_axis"})
    )

    df4_lb = (
        data4.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_lb, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_lb_fixed_axis"})
    )
    df4_ub = (
        data4.groupby(list_cols_cohort)[fixed_axis]
        .quantile(numeric_only=True, q=q_ub, interpolation="nearest")
        .reset_index()
        .rename(columns={fixed_axis: "_ub_fixed_axis"})
    )
    # Merge back
    df1 = df1.merge(df1_lb, on=list_cols_cohort, how="left")
    df1 = df1.merge(df1_ub, on=list_cols_cohort, how="left")
    df2 = df2.merge(df2_lb, on=list_cols_cohort, how="left")
    df2 = df2.merge(df2_ub, on=list_cols_cohort, how="left")
    df3 = df3.merge(df3_lb, on=list_cols_cohort, how="left")
    df3 = df3.merge(df3_ub, on=list_cols_cohort, how="left")
    df4 = df4.merge(df4_lb, on=list_cols_cohort, how="left")
    df4 = df4.merge(df4_ub, on=list_cols_cohort, how="left")
    # Keep rows where fixed axis observations fall within range
    df1 = df1[
        (df1[fixed_axis] >= df1["_lb_fixed_axis"])
        & (df1[fixed_axis] <= df1["_ub_fixed_axis"])
    ]
    df2 = df2[
        (df2[fixed_axis] >= df2["_lb_fixed_axis"])
        & (df2[fixed_axis] <= df2["_ub_fixed_axis"])
    ]
    df3 = df3[
        (df3[fixed_axis] >= df3["_lb_fixed_axis"])
        & (df3[fixed_axis] <= df3["_ub_fixed_axis"])
    ]
    df4 = df4[
        (df4[fixed_axis] >= df4["_lb_fixed_axis"])
        & (df4[fixed_axis] <= df4["_ub_fixed_axis"])
    ]
    # Collapse
    df1_agg = (
        df1.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df2_agg = (
        df2.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df3_agg = (
        df3.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    df4_agg = (
        df4.groupby(list_cols_cohort)[list_cols_outcomes]
        .mean(numeric_only=True)
        .reset_index()
    )
    # Generate time identifiers
    df1_agg["_time"] = 1
    df2_agg["_time"] = 2
    df3_agg["_time"] = 3
    df4_agg["_time"] = 4
    # Merge (unbalanced)
    df_agg = pd.concat([df1_agg, df2_agg, df3_agg, df4_agg], axis=0)
    df_agg = df_agg.sort_values(by=list_cols_cohort + ["_time"]).reset_index(drop=True)
    # Merge (balanced)
    groups_balanced = df1_agg[list_cols_cohort].merge(
        df2_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df3_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced = groups_balanced[list_cols_cohort].merge(
        df4_agg[list_cols_cohort], on=list_cols_cohort, how="inner"
    )
    groups_balanced["balanced"] = 1
    df_agg_balanced = df_agg.merge(groups_balanced, on=list_cols_cohort, how="left")
    df_agg_balanced = df_agg_balanced[df_agg_balanced["balanced"] == 1]
    del df_agg_balanced["balanced"]
    df_agg_balanced = df_agg_balanced.sort_values(
        by=list_cols_cohort + ["_time"]
    ).reset_index(drop=True)
    
    # calculate number of individual observations per cohort
    obscohort_df1 = data1.groupby(list_cols_cohort).size()
    obscohort_df2 = data2.groupby(list_cols_cohort).size()
    obscohort_df3 = data1.groupby(list_cols_cohort).size()
    obscohort_df4 = data1.groupby(list_cols_cohort).size()
    obscohort_all = pd.concat(
        [obscohort_df1, obscohort_df2, obscohort_df3, obscohort_df4], axis=0
    )
    obscohort = pd.DataFrame(
        {
            "Year": ["2014", "2016", "2019", "2022", "Total"],
            "Avg Obs Per Cohort": [
                len(data1) / len(df1_agg),
                len(data2) / len(df2_agg),
                len(data3) / len(df3_agg),
                len(data4) / len(df4_agg),
                (len(data1) + len(data2) + len(data3) + len(data4)) / len(df_agg_balanced) / 4,
            ],
            "Number of Cohorts": [
                len(df1_agg),
                len(df2_agg),
                len(df3_agg),
                len(df4_agg),
                len(df_agg_balanced) / 4,
            ],
        }
    )
    # Save to local
    df_agg.to_parquet(base_path + "hies_consol_agg_" + file_suffix + ".parquet")
    df_agg_balanced.to_parquet(
        base_path + "hies_consol_agg_balanced_" + file_suffix + ".parquet"
    )
    obscohort.to_parquet(
        base_path + "hies_consol_agg_obscohort_" + file_suffix + ".parquet"
    )
    # Output
    return df_agg, df_agg_balanced


# Check value overlaps
for col in col_groups:
    print("[2014] " + col + ": " + ", ".join(list(df_14[col].astype("str").unique())))
    print("[2016] " + col + ": " + ", ".join(list(df_16[col].astype("str").unique())))
    print("[2019] " + col + ": " + ", ".join(list(df_19[col].astype("str").unique())))
    print("[2022] " + col + ": " + ", ".join(list(df_22[col].astype("str").unique())))

# The merging part
df_agg_mean, df_agg_balanced_mean = gen_pseudopanel(
    data1=df_14,
    data2=df_16,
    data3=df_19,
    data4=df_22,
    list_cols_cohort=col_groups,
    list_cols_outcomes=col_outcomes,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix="mean",
    base_path=path_data,
)

df_agg_mean_hhbasis, df_agg_balanced_mean_hhbasis = gen_pseudopanel(
    data1=df_14_hhbasis,
    data2=df_16_hhbasis,
    data3=df_19_hhbasis,
    data4=df_22_hhbasis,
    list_cols_cohort=col_groups_with_hhsize,
    list_cols_outcomes=col_outcomes,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix="mean_hhbasis",
    base_path=path_data,
)

df_agg_mean_equivalised, df_agg_balanced_mean_equivalised = gen_pseudopanel(
    data1=df_14_equivalised,
    data2=df_16_equivalised,
    data3=df_19_equivalised,
    data4=df_22_equivalised,
    list_cols_cohort=col_groups,
    list_cols_outcomes=col_outcomes,
    use_mean=True,
    use_quantile=False,
    quantile_choice=0.5,
    file_suffix="mean_equivalised",
    base_path=path_data,
)

list_distbounds = [[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]]
list_suffixes = ["20p", "40p", "60p", "80p", "100p"]
# list_suffixes = ['10p', '20p', '30p', '40p', '50p', '60p', '70p', '80p', '90p', '100p']
for distbound, suffix in tqdm(zip(list_distbounds, list_suffixes)):
    # Use fixed axis on gross income when generating 10pp width buckets cohort panel
    df_agg_quantile, df_agg_balanced_quantile = gen_pseudopanel_distgroup_fixed_axis(
        data1=df_14,
        data2=df_16,
        data3=df_19,
        data4=df_22,
        fixed_axis="gross_income",
        list_cols_cohort=col_groups,
        list_cols_outcomes=col_outcomes,
        distbounds=distbound,
        file_suffix=suffix,
        base_path=path_data,
    )
    (
        df_agg_quantile_hhbasis,
        df_agg_balanced_quantile_hhbasis,
    ) = gen_pseudopanel_distgroup_fixed_axis(
        data1=df_14_hhbasis,
        data2=df_16_hhbasis,
        data3=df_19_hhbasis,
        data4=df_22_hhbasis,
        fixed_axis="gross_income",
        list_cols_cohort=col_groups_with_hhsize,
        list_cols_outcomes=col_outcomes,
        distbounds=distbound,
        file_suffix=suffix + "_hhbasis",
        base_path=path_data,
    )
    (
        df_agg_quantile_equivalised,
        df_agg_balanced_quantile_equivalised,
    ) = gen_pseudopanel_distgroup_fixed_axis(
        data1=df_14_equivalised,
        data2=df_16_equivalised,
        data3=df_19_equivalised,
        data4=df_22_equivalised,
        fixed_axis="gross_income",
        list_cols_cohort=col_groups_with_hhsize,
        list_cols_outcomes=col_outcomes,
        distbounds=distbound,
        file_suffix=suffix + "_equivalised",
        base_path=path_data,
    )

# Individual pooled data + output
df_14["year"] = 2014
df_16["year"] = 2016
df_19["year"] = 2019
df_22["year"] = 2022
df_ind = pd.concat([df_14, df_16, df_19, df_22], axis=0)
df_ind = df_ind.sort_values(by=col_groups + ["year"]).reset_index(drop=True)
df_ind.to_parquet(path_data + "hies_consol_ind.parquet")

# Individual pooled data + output (household basis)
df_14_hhbasis["year"] = 2014
df_16_hhbasis["year"] = 2016
df_19_hhbasis["year"] = 2019
df_22_hhbasis["year"] = 2022
df_ind_hhbasis = pd.concat(
    [df_14_hhbasis, df_16_hhbasis, df_19_hhbasis, df_22_hhbasis], axis=0
)
df_ind_hhbasis = df_ind_hhbasis.sort_values(by=col_groups + ["year"]).reset_index(
    drop=True
)
df_ind_hhbasis.to_parquet(path_data + "hies_consol_ind_hhbasis.parquet")

# Individual pooled data + output (equivalised)
df_14_equivalised["year"] = 2014
df_16_equivalised["year"] = 2016
df_19_equivalised["year"] = 2019
df_22_equivalised["year"] = 2022
df_ind_equivalised = pd.concat(
    [df_14_equivalised, df_16_equivalised, df_19_equivalised, df_22_equivalised], axis=0
)
df_ind_equivalised = df_ind_equivalised.sort_values(
    by=col_groups + ["year"]
).reset_index(drop=True)
df_ind_equivalised.to_parquet(path_data + "hies_consol_ind_equivalised.parquet")

# Individual pooled data + output (with outliers)
df_14_full["year"] = 2014
df_16_full["year"] = 2016
df_19_full["year"] = 2019
df_22_full["year"] = 2022
df_ind_full = pd.concat([df_14_full, df_16_full, df_19_full, df_22_full], axis=0)
df_ind_full = df_ind_full.sort_values(by=col_groups + ["year"]).reset_index(drop=True)
df_ind_full.to_parquet(path_data + "hies_consol_ind_full.parquet")

# Individual pooled data + output (with outliers and household basis)
df_14_full_hhbasis["year"] = 2014
df_16_full_hhbasis["year"] = 2016
df_19_full_hhbasis["year"] = 2019
df_22_full_hhbasis["year"] = 2022
df_ind_full_hhbasis = pd.concat(
    [df_14_full_hhbasis, df_16_full_hhbasis, df_19_full_hhbasis, df_22_full_hhbasis],
    axis=0,
)
df_ind_full_hhbasis = df_ind_full_hhbasis.sort_values(
    by=col_groups + ["year"]
).reset_index(drop=True)
df_ind_full_hhbasis.to_parquet(path_data + "hies_consol_ind_full_hhbasis.parquet")

# Individual pooled data + output (with outliers and equivalised)
df_14_full_equivalised["year"] = 2014
df_16_full_equivalised["year"] = 2016
df_19_full_equivalised["year"] = 2019
df_22_full_equivalised["year"] = 2022
df_ind_full_equivalised = pd.concat(
    [
        df_14_full_equivalised,
        df_16_full_equivalised,
        df_19_full_equivalised,
        df_22_full_equivalised,
    ],
    axis=0,
)
df_ind_full_equivalised = df_ind_full_equivalised.sort_values(
    by=col_groups + ["year"]
).reset_index(drop=True)
df_ind_full_equivalised.to_parquet(
    path_data + "hies_consol_ind_full_equivalised.parquet"
)

# %%
# IV.B --------------------- Merger (group-level; 2014 - 2022; for LIFECYCLE analyses) ---------------------
# Remove years
del df_14["year"]
del df_16["year"]
del df_19["year"]
del df_22["year"]
# Set up loop
print_balanced = True
list_incgroups = ["0_b20-", "1_b20+", "2_m20-", "3_m20+", "4_t20"]
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_birth_year_groups = list(df_22["birth_year_group"].unique())
for incgroup, incgroup_suffix in zip(list_incgroups, list_incgroups_suffix):
    for birth_year_group in tqdm(list_birth_year_groups):
        df_14_sub_hhbasis = df_14_hhbasis_withbase[
            (
                (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_14_hhbasis_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_16_sub_hhbasis = df_16_hhbasis_withbase[
            (
                (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_16_hhbasis_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_19_sub_hhbasis = df_19_hhbasis_withbase[
            (
                (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_19_hhbasis_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_22_sub_hhbasis = df_22_hhbasis_withbase[
            (
                (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_22_hhbasis_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()

        # equivalised
        df_14_sub_equivalised = df_14_equivalised_withbase[
            (
                (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_14_equivalised_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_16_sub_equivalised = df_16_equivalised_withbase[
            (
                (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_16_equivalised_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_19_sub_equivalised = df_19_equivalised_withbase[
            (
                (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_19_equivalised_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_22_sub_equivalised = df_22_equivalised_withbase[
            (
                (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_22_equivalised_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()

        # per capita
        df_14_sub = df_14_withbase[
            (
                (df_14_withbase["gross_income_group"] == incgroup)
                & (df_14_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_16_sub = df_16_withbase[
            (
                (df_16_withbase["gross_income_group"] == incgroup)
                & (df_16_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_19_sub = df_19_withbase[
            (
                (df_19_withbase["gross_income_group"] == incgroup)
                & (df_19_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()
        df_22_sub = df_22_withbase[
            (
                (df_22_withbase["gross_income_group"] == incgroup)
                & (df_22_withbase["birth_year_group"] == birth_year_group)
            )
        ].copy()

        # Delete variables that already have base variables created
        for i in cols_base_group_transformed:
            del df_14_sub[i]
            del df_16_sub[i]
            del df_19_sub[i]
            del df_22_sub[i]

            del df_14_sub_equivalised[i]
            del df_16_sub_equivalised[i]
            del df_19_sub_equivalised[i]
            del df_22_sub_equivalised[i]

        for i in cols_base_group_transformed_with_hhsize:
            del df_14_sub_hhbasis[i]
            del df_16_sub_hhbasis[i]
            del df_19_sub_hhbasis[i]
            del df_22_sub_hhbasis[i]

        # Compile pseudo panel data
        df_agg_mean_sub, df_agg_balanced_mean_sub = gen_pseudopanel(
            data1=df_14_sub,
            data2=df_16_sub,
            data3=df_19_sub,
            data4=df_22_sub,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_" + birth_year_group,
            base_path=path_subgroup,
        )
        df_agg_mean_sub_hhbasis, df_agg_balanced_mean_sub_hhbasis = gen_pseudopanel(
            data1=df_14_sub_hhbasis,
            data2=df_16_sub_hhbasis,
            data3=df_19_sub_hhbasis,
            data4=df_22_sub_hhbasis,
            list_cols_cohort=col_groups_with_hhsize,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_" + birth_year_group + "_hhbasis",
            base_path=path_subgroup,
        )
        (
            df_agg_mean_sub_equivalised,
            df_agg_balanced_mean_sub_equivalised,
        ) = gen_pseudopanel(
            data1=df_14_sub_equivalised,
            data2=df_16_sub_equivalised,
            data3=df_19_sub_equivalised,
            data4=df_22_sub_equivalised,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_"
            + incgroup_suffix
            + "_"
            + birth_year_group
            + "_equivalised",
            base_path=path_subgroup,
        )
        # Print out length of dataframe
        if print_balanced:
            print(
                "Number of balanced cohorts for "
                + incgroup
                + " with birth years in "
                + birth_year_group
                + ": "
                + str(int(len(df_agg_balanced_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )
        elif not print_balanced:
            print(
                "Number of cohorts (balanced / unbalanced) for "
                + incgroup
                + " with birth years in "
                + birth_year_group
                + ": "
                + str(int(len(df_agg_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )


# %%
# IV.C --------------------- Merger (group-level; 2014 - 2022; for SUBGROUP analyses) ---------------------
# Reduced set of col_groups
# Create subgroups by benefit objective B (kid, working_age2, and elderly2), and income groups Y
list_incgroups = ["0_b20-", "1_b20+", "2_m20-", "3_m20+", "4_t20"]
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_benefit_objectives = ["kid", "working_age2", "elderly2"]
list_truncate_thresholds = [
    1,
    1,
    1,
]  # [2, 2, 2] [3, 2, 1] [3, 2, 2] [4, 3, 2] [2, 2, 1]
print_balanced = True
for incgroup, incgroup_suffix in zip(list_incgroups, list_incgroups_suffix):
    for benefit_objective, truncate_threshold in zip(
        list_benefit_objectives, list_truncate_thresholds
    ):
        for n_obj in range(1, truncate_threshold + 1):
            if n_obj < truncate_threshold:
                # for file suffix
                n_obj_nice = str(n_obj)

                # hh basis
                df_14_sub_hhbasis = df_14_hhbasis_withbase[
                    (
                        (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_14_hhbasis_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_16_sub_hhbasis = df_16_hhbasis_withbase[
                    (
                        (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_16_hhbasis_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_19_sub_hhbasis = df_19_hhbasis_withbase[
                    (
                        (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_19_hhbasis_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_22_sub_hhbasis = df_22_hhbasis_withbase[
                    (
                        (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_22_hhbasis_withbase[benefit_objective] == n_obj)
                    )
                ].copy()

                # equivalised
                df_14_sub_equivalised = df_14_equivalised_withbase[
                    (
                        (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_14_equivalised_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_16_sub_equivalised = df_16_equivalised_withbase[
                    (
                        (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_16_equivalised_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_19_sub_equivalised = df_19_equivalised_withbase[
                    (
                        (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_19_equivalised_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_22_sub_equivalised = df_22_equivalised_withbase[
                    (
                        (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_22_equivalised_withbase[benefit_objective] == n_obj)
                    )
                ].copy()

                # per capita
                df_14_sub = df_14_withbase[
                    (
                        (df_14_withbase["gross_income_group"] == incgroup)
                        & (df_14_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_16_sub = df_16_withbase[
                    (
                        (df_16_withbase["gross_income_group"] == incgroup)
                        & (df_16_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_19_sub = df_19_withbase[
                    (
                        (df_19_withbase["gross_income_group"] == incgroup)
                        & (df_19_withbase[benefit_objective] == n_obj)
                    )
                ].copy()
                df_22_sub = df_22_withbase[
                    (
                        (df_22_withbase["gross_income_group"] == incgroup)
                        & (df_22_withbase[benefit_objective] == n_obj)
                    )
                ].copy()

            if n_obj == truncate_threshold:
                # for file suffix
                n_obj_nice = str(n_obj) + "plus"

                # hh basis
                df_14_sub_hhbasis = df_14_hhbasis_withbase[
                    (
                        (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_14_hhbasis_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_16_sub_hhbasis = df_16_hhbasis_withbase[
                    (
                        (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_16_hhbasis_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_19_sub_hhbasis = df_19_hhbasis_withbase[
                    (
                        (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_19_hhbasis_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_22_sub_hhbasis = df_22_hhbasis_withbase[
                    (
                        (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                        & (df_22_hhbasis_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()

                # equivalised
                df_14_sub_equivalised = df_14_equivalised_withbase[
                    (
                        (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_14_equivalised_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_16_sub_equivalised = df_16_equivalised_withbase[
                    (
                        (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_16_equivalised_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_19_sub_equivalised = df_19_equivalised_withbase[
                    (
                        (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_19_equivalised_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_22_sub_equivalised = df_22_equivalised_withbase[
                    (
                        (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                        & (df_22_equivalised_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()

                # per capita
                df_14_sub = df_14_withbase[
                    (
                        (df_14_withbase["gross_income_group"] == incgroup)
                        & (df_14_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_16_sub = df_16_withbase[
                    (
                        (df_16_withbase["gross_income_group"] == incgroup)
                        & (df_16_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_19_sub = df_19_withbase[
                    (
                        (df_19_withbase["gross_income_group"] == incgroup)
                        & (df_19_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()
                df_22_sub = df_22_withbase[
                    (
                        (df_22_withbase["gross_income_group"] == incgroup)
                        & (df_22_withbase[benefit_objective] >= n_obj)
                    )
                ].copy()

            # Delete variables that already have base variables created
            for i in cols_base_group_transformed:
                del df_14_sub[i]
                del df_16_sub[i]
                del df_19_sub[i]
                del df_22_sub[i]

                del df_14_sub_equivalised[i]
                del df_16_sub_equivalised[i]
                del df_19_sub_equivalised[i]
                del df_22_sub_equivalised[i]

            for i in cols_base_group_transformed_with_hhsize:
                del df_14_sub_hhbasis[i]
                del df_16_sub_hhbasis[i]
                del df_19_sub_hhbasis[i]
                del df_22_sub_hhbasis[i]

            # Compile pseudo panel data
            df_agg_mean_sub, df_agg_balanced_mean_sub = gen_pseudopanel(
                data1=df_14_sub,
                data2=df_16_sub,
                data3=df_19_sub,
                data4=df_22_sub,
                list_cols_cohort=col_groups,
                list_cols_outcomes=col_outcomes,
                use_mean=True,
                use_quantile=False,
                quantile_choice=0.5,
                file_suffix="mean_"
                + incgroup_suffix
                + "_"
                + benefit_objective
                + "_"
                + n_obj_nice,
                base_path=path_subgroup,
            )
            df_agg_mean_sub_hhbasis, df_agg_balanced_mean_sub_hhbasis = gen_pseudopanel(
                data1=df_14_sub_hhbasis,
                data2=df_16_sub_hhbasis,
                data3=df_19_sub_hhbasis,
                data4=df_22_sub_hhbasis,
                list_cols_cohort=col_groups_with_hhsize,
                list_cols_outcomes=col_outcomes,
                use_mean=True,
                use_quantile=False,
                quantile_choice=0.5,
                file_suffix="mean_"
                + incgroup_suffix
                + "_"
                + benefit_objective
                + "_"
                + n_obj_nice
                + "_hhbasis",
                base_path=path_subgroup,
            )
            (
                df_agg_mean_sub_equivalised,
                df_agg_balanced_mean_sub_equivalised,
            ) = gen_pseudopanel(
                data1=df_14_sub_equivalised,
                data2=df_16_sub_equivalised,
                data3=df_19_sub_equivalised,
                data4=df_22_sub_equivalised,
                list_cols_cohort=col_groups,
                list_cols_outcomes=col_outcomes,
                use_mean=True,
                use_quantile=False,
                quantile_choice=0.5,
                file_suffix="mean_"
                + incgroup_suffix
                + "_"
                + benefit_objective
                + "_"
                + n_obj_nice
                + "_equivalised",
                base_path=path_subgroup,
            )

            # Print out length of dataframe
            if print_balanced:
                print(
                    "Number of balanced cohorts for "
                    + incgroup
                    + " with "
                    + n_obj_nice
                    + " "
                    + benefit_objective
                    + ": "
                    + str(int(len(df_agg_balanced_mean_sub_hhbasis) / 4))
                    + " (from "
                    + str(
                        int(
                            np.amin(
                                [
                                    len(df_14_sub_hhbasis),
                                    len(df_16_sub_hhbasis),
                                    len(df_19_sub_hhbasis),
                                    len(df_22_sub_hhbasis),
                                ]
                            )
                        )
                    )
                    + " households)"
                )
            elif not print_balanced:
                print(
                    "Number of cohorts (balanced / unbalanced) for "
                    + incgroup
                    + " with "
                    + n_obj_nice
                    + " "
                    + benefit_objective
                    + ": "
                    + str(int(len(df_agg_mean_sub_hhbasis) / 4))
                    + " (from "
                    + str(
                        int(
                            np.amin(
                                [
                                    len(df_14_sub_hhbasis),
                                    len(df_16_sub_hhbasis),
                                    len(df_19_sub_hhbasis),
                                    len(df_22_sub_hhbasis),
                                ]
                            )
                        )
                    )
                    + " households)"
                )


# IV.D --------------------- Merger (group-level; 2014 - 2022; for ETHNICITY analyses) ---------------------
# Set up loop
print_balanced = True
list_incgroups = ["0_b20-", "1_b20+", "2_m20-", "3_m20+", "4_t20"]
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_ethnicities = list(df_22["ethnicity"].unique())
for incgroup, incgroup_suffix in zip(list_incgroups, list_incgroups_suffix):
    for ethnicity in tqdm(list_ethnicities):
        df_14_sub_hhbasis = df_14_hhbasis_withbase[
            (
                (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_14_hhbasis_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_16_sub_hhbasis = df_16_hhbasis_withbase[
            (
                (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_16_hhbasis_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_19_sub_hhbasis = df_19_hhbasis_withbase[
            (
                (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_19_hhbasis_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_22_sub_hhbasis = df_22_hhbasis_withbase[
            (
                (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_22_hhbasis_withbase["ethnicity"] == ethnicity)
            )
        ].copy()

        # equivalised
        df_14_sub_equivalised = df_14_equivalised_withbase[
            (
                (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_14_equivalised_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_16_sub_equivalised = df_16_equivalised_withbase[
            (
                (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_16_equivalised_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_19_sub_equivalised = df_19_equivalised_withbase[
            (
                (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_19_equivalised_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_22_sub_equivalised = df_22_equivalised_withbase[
            (
                (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_22_equivalised_withbase["ethnicity"] == ethnicity)
            )
        ].copy()

        # per capita
        df_14_sub = df_14_withbase[
            (
                (df_14_withbase["gross_income_group"] == incgroup)
                & (df_14_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_16_sub = df_16_withbase[
            (
                (df_16_withbase["gross_income_group"] == incgroup)
                & (df_16_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_19_sub = df_19_withbase[
            (
                (df_19_withbase["gross_income_group"] == incgroup)
                & (df_19_withbase["ethnicity"] == ethnicity)
            )
        ].copy()
        df_22_sub = df_22_withbase[
            (
                (df_22_withbase["gross_income_group"] == incgroup)
                & (df_22_withbase["ethnicity"] == ethnicity)
            )
        ].copy()

        # Delete variables that already have base variables created
        for i in cols_base_group_transformed:
            del df_14_sub[i]
            del df_16_sub[i]
            del df_19_sub[i]
            del df_22_sub[i]

            del df_14_sub_equivalised[i]
            del df_16_sub_equivalised[i]
            del df_19_sub_equivalised[i]
            del df_22_sub_equivalised[i]

        for i in cols_base_group_transformed_with_hhsize:
            del df_14_sub_hhbasis[i]
            del df_16_sub_hhbasis[i]
            del df_19_sub_hhbasis[i]
            del df_22_sub_hhbasis[i]

        # Compile pseudo panel data
        df_agg_mean_sub, df_agg_balanced_mean_sub = gen_pseudopanel(
            data1=df_14_sub,
            data2=df_16_sub,
            data3=df_19_sub,
            data4=df_22_sub,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_" + ethnicity,
            base_path=path_subgroup,
        )
        df_agg_mean_sub_hhbasis, df_agg_balanced_mean_sub_hhbasis = gen_pseudopanel(
            data1=df_14_sub_hhbasis,
            data2=df_16_sub_hhbasis,
            data3=df_19_sub_hhbasis,
            data4=df_22_sub_hhbasis,
            list_cols_cohort=col_groups_with_hhsize,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_" + ethnicity + "_hhbasis",
            base_path=path_subgroup,
        )
        (
            df_agg_mean_sub_equivalised,
            df_agg_balanced_mean_sub_equivalised,
        ) = gen_pseudopanel(
            data1=df_14_sub_equivalised,
            data2=df_16_sub_equivalised,
            data3=df_19_sub_equivalised,
            data4=df_22_sub_equivalised,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_" + ethnicity + "_equivalised",
            base_path=path_subgroup,
        )
        # Print out length of dataframe
        if print_balanced:
            print(
                "Number of balanced cohorts for "
                + incgroup
                + " of ethnic group "
                + ethnicity
                + ": "
                + str(int(len(df_agg_balanced_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )
        elif not print_balanced:
            print(
                "Number of cohorts (balanced / unbalanced) for "
                + incgroup
                + " of ethnic group "
                + ethnicity
                + ": "
                + str(int(len(df_agg_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )



# IV.E --------------------- Merger (group-level; 2014 - 2022; for STATE analyses) ---------------------
# Set up loop
print_balanced = True
list_incgroups = ["0_b20-", "1_b20+", "2_m20-", "3_m20+", "4_t20"]
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_urban_rural = list(df_22["urban"].unique())
for incgroup, incgroup_suffix in zip(list_incgroups, list_incgroups_suffix):
    for urban in tqdm(list_urban_rural):
        print("\n" + str(urban) + ", " + incgroup)
        df_14_sub_hhbasis = df_14_hhbasis_withbase[
            (
                (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_14_hhbasis_withbase["urban"] == urban)
            )
        ].copy()
        df_16_sub_hhbasis = df_16_hhbasis_withbase[
            (
                (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_16_hhbasis_withbase["urban"] == urban)
            )
        ].copy()
        df_19_sub_hhbasis = df_19_hhbasis_withbase[
            (
                (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_19_hhbasis_withbase["urban"] == urban)
            )
        ].copy()
        df_22_sub_hhbasis = df_22_hhbasis_withbase[
            (
                (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_22_hhbasis_withbase["urban"] == urban)
            )
        ].copy()

        # equivalised
        df_14_sub_equivalised = df_14_equivalised_withbase[
            (
                (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_14_equivalised_withbase["urban"] == urban)
            )
        ].copy()
        df_16_sub_equivalised = df_16_equivalised_withbase[
            (
                (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_16_equivalised_withbase["urban"] == urban)
            )
        ].copy()
        df_19_sub_equivalised = df_19_equivalised_withbase[
            (
                (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_19_equivalised_withbase["urban"] == urban)
            )
        ].copy()
        df_22_sub_equivalised = df_22_equivalised_withbase[
            (
                (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_22_equivalised_withbase["urban"] == urban)
            )
        ].copy()

        # per capita
        df_14_sub = df_14_withbase[
            (
                (df_14_withbase["gross_income_group"] == incgroup)
                & (df_14_withbase["urban"] == urban)
            )
        ].copy()
        df_16_sub = df_16_withbase[
            (
                (df_16_withbase["gross_income_group"] == incgroup)
                & (df_16_withbase["urban"] == urban)
            )
        ].copy()
        df_19_sub = df_19_withbase[
            (
                (df_19_withbase["gross_income_group"] == incgroup)
                & (df_19_withbase["urban"] == urban)
            )
        ].copy()
        df_22_sub = df_22_withbase[
            (
                (df_22_withbase["gross_income_group"] == incgroup)
                & (df_22_withbase["urban"] == urban)
            )
        ].copy()

        # Delete variables that already have base variables created
        for i in cols_base_group_transformed:
            del df_14_sub[i]
            del df_16_sub[i]
            del df_19_sub[i]
            del df_22_sub[i]

            del df_14_sub_equivalised[i]
            del df_16_sub_equivalised[i]
            del df_19_sub_equivalised[i]
            del df_22_sub_equivalised[i]

        for i in cols_base_group_transformed_with_hhsize:
            del df_14_sub_hhbasis[i]
            del df_16_sub_hhbasis[i]
            del df_19_sub_hhbasis[i]
            del df_22_sub_hhbasis[i]

        # Compile pseudo panel data
        df_agg_mean_sub, df_agg_balanced_mean_sub = gen_pseudopanel(
            data1=df_14_sub,
            data2=df_16_sub,
            data3=df_19_sub,
            data4=df_22_sub,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_urban" + str(urban),
            base_path=path_subgroup,
        )
        df_agg_mean_sub_hhbasis, df_agg_balanced_mean_sub_hhbasis = gen_pseudopanel(
            data1=df_14_sub_hhbasis,
            data2=df_16_sub_hhbasis,
            data3=df_19_sub_hhbasis,
            data4=df_22_sub_hhbasis,
            list_cols_cohort=col_groups_with_hhsize,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_urban" + str(urban) + "_hhbasis",
            base_path=path_subgroup,
        )
        (
            df_agg_mean_sub_equivalised,
            df_agg_balanced_mean_sub_equivalised,
        ) = gen_pseudopanel(
            data1=df_14_sub_equivalised,
            data2=df_16_sub_equivalised,
            data3=df_19_sub_equivalised,
            data4=df_22_sub_equivalised,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_urban" + str(urban) + "_equivalised",
            base_path=path_subgroup,
        )
        # Print out length of dataframe
        if print_balanced:
            print(
                "Number of balanced cohorts for "
                + incgroup
                + " of strata "
                + str(urban)
                + ": "
                + str(int(len(df_agg_balanced_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )
        elif not print_balanced:
            print(
                "Number of cohorts (balanced / unbalanced) for "
                + incgroup
                + " of strata "
                + str(urban)
                + ": "
                + str(int(len(df_agg_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )


# IV.F --------------------- Merger (group-level; 2014 - 2022; for GENDER analyses) ---------------------
# Set up loop
print_balanced = True
list_incgroups = ["0_b20-", "1_b20+", "2_m20-", "3_m20+", "4_t20"]
list_incgroups_suffix = ["b20m", "b20p", "m20m", "m20p", "t20"]
list_gender = list(df_22["male"].unique())
for incgroup, incgroup_suffix in zip(list_incgroups, list_incgroups_suffix):
    for male in tqdm(list_gender):
        print("\n" + str(male) + ", " + incgroup)
        df_14_sub_hhbasis = df_14_hhbasis_withbase[
            (
                (df_14_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_14_hhbasis_withbase["male"] == male)
            )
        ].copy()
        df_16_sub_hhbasis = df_16_hhbasis_withbase[
            (
                (df_16_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_16_hhbasis_withbase["male"] == male)
            )
        ].copy()
        df_19_sub_hhbasis = df_19_hhbasis_withbase[
            (
                (df_19_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_19_hhbasis_withbase["male"] == male)
            )
        ].copy()
        df_22_sub_hhbasis = df_22_hhbasis_withbase[
            (
                (df_22_hhbasis_withbase["gross_income_group"] == incgroup)
                & (df_22_hhbasis_withbase["male"] == male)
            )
        ].copy()

        # equivalised
        df_14_sub_equivalised = df_14_equivalised_withbase[
            (
                (df_14_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_14_equivalised_withbase["male"] == male)
            )
        ].copy()
        df_16_sub_equivalised = df_16_equivalised_withbase[
            (
                (df_16_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_16_equivalised_withbase["male"] == male)
            )
        ].copy()
        df_19_sub_equivalised = df_19_equivalised_withbase[
            (
                (df_19_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_19_equivalised_withbase["male"] == male)
            )
        ].copy()
        df_22_sub_equivalised = df_22_equivalised_withbase[
            (
                (df_22_equivalised_withbase["gross_income_group"] == incgroup)
                & (df_22_equivalised_withbase["male"] == male)
            )
        ].copy()

        # per capita
        df_14_sub = df_14_withbase[
            (
                (df_14_withbase["gross_income_group"] == incgroup)
                & (df_14_withbase["male"] == male)
            )
        ].copy()
        df_16_sub = df_16_withbase[
            (
                (df_16_withbase["gross_income_group"] == incgroup)
                & (df_16_withbase["male"] == male)
            )
        ].copy()
        df_19_sub = df_19_withbase[
            (
                (df_19_withbase["gross_income_group"] == incgroup)
                & (df_19_withbase["male"] == male)
            )
        ].copy()
        df_22_sub = df_22_withbase[
            (
                (df_22_withbase["gross_income_group"] == incgroup)
                & (df_22_withbase["male"] == male)
            )
        ].copy()

        # Delete variables that already have base variables created
        for i in cols_base_group_transformed:
            del df_14_sub[i]
            del df_16_sub[i]
            del df_19_sub[i]
            del df_22_sub[i]

            del df_14_sub_equivalised[i]
            del df_16_sub_equivalised[i]
            del df_19_sub_equivalised[i]
            del df_22_sub_equivalised[i]

        for i in cols_base_group_transformed_with_hhsize:
            del df_14_sub_hhbasis[i]
            del df_16_sub_hhbasis[i]
            del df_19_sub_hhbasis[i]
            del df_22_sub_hhbasis[i]

        # Compile pseudo panel data
        df_agg_mean_sub, df_agg_balanced_mean_sub = gen_pseudopanel(
            data1=df_14_sub,
            data2=df_16_sub,
            data3=df_19_sub,
            data4=df_22_sub,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_male" + str(male),
            base_path=path_subgroup,
        )
        df_agg_mean_sub_hhbasis, df_agg_balanced_mean_sub_hhbasis = gen_pseudopanel(
            data1=df_14_sub_hhbasis,
            data2=df_16_sub_hhbasis,
            data3=df_19_sub_hhbasis,
            data4=df_22_sub_hhbasis,
            list_cols_cohort=col_groups_with_hhsize,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_male" + str(male) + "_hhbasis",
            base_path=path_subgroup,
        )
        (
            df_agg_mean_sub_equivalised,
            df_agg_balanced_mean_sub_equivalised,
        ) = gen_pseudopanel(
            data1=df_14_sub_equivalised,
            data2=df_16_sub_equivalised,
            data3=df_19_sub_equivalised,
            data4=df_22_sub_equivalised,
            list_cols_cohort=col_groups,
            list_cols_outcomes=col_outcomes,
            use_mean=True,
            use_quantile=False,
            quantile_choice=0.5,
            file_suffix="mean_" + incgroup_suffix + "_male" + str(male) + "_equivalised",
            base_path=path_subgroup,
        )
        # Print out length of dataframe
        if print_balanced:
            print(
                "Number of balanced cohorts for "
                + incgroup
                + " of gender of head of HH "
                + str(male)
                + ": "
                + str(int(len(df_agg_balanced_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )
        elif not print_balanced:
            print(
                "Number of cohorts (balanced / unbalanced) for "
                + incgroup
                + " of gender of head of HH "
                + str(male)
                + ": "
                + str(int(len(df_agg_mean_sub_hhbasis) / 4))
                + " (from "
                + str(
                    int(
                        np.amin(
                            [
                                len(df_14_sub_hhbasis),
                                len(df_16_sub_hhbasis),
                                len(df_19_sub_hhbasis),
                                len(df_22_sub_hhbasis),
                            ]
                        )
                    )
                )
                + " households)"
            )

# %%
# X --- Notify
# telsendmsg(conf=tel_config, msg="impact-household --- process_consol_group: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
