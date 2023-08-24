# %%
import pandas as pd
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
path_data = "data/hies_2022/"  # 2022

# %%
# I --- Load part 1 (diff from previous vintages)
df_p1 = pd.read_spss(path_data + "HES 2022- Member-MOF.sav")  # individual basis
cols_df_p1 = pd.Series(df_p1.columns)
df_p1

# %%
# II --- Load part 2 (diff from previous vintages)
df_p2 = pd.read_spss(path_data + "2022- Exp HH- MOF.sav")  # hh basis
cols_df_p2 = pd.Series(df_p2.columns)
df_p2

# %%
# III --- Load income only (diff from previous vintages)
df_inc = pd.read_excel(
    path_data + "HIS 2022_household (n=87411).xlsx", sheet_name="Sheet1"
)  # individual basis
cols_df_inc = pd.Series(df_inc.columns)
df_inc

# %%
# IV --- Clean & Consolidate
# %%
# Convert categorical columns into str
cols_cat = df_p2.select_dtypes(include=["category"]).columns
df_p2[cols_cat] = df_p2[cols_cat].astype(str)

# %%
# Harmonise column labels (to p2; HH basis)
dict_col_harmonise = {
    "Relationship": "Relationship",
    "Gender": "Jantina",
    "Age": "Umur",
    "Citizenship": "Kewarganegaraan",
    "Marital_status": "Taraf_perkahwinan",
    "Inc_recipient": "Penerima_pendapatan",
    "State": "Negeri",
    "Strata": "Strata",
    "Saiz_hh": "Saiz_isirumah",
    "Ethnic": "Kaum",
    "Highest_Certificate_Obt": "Sijil_tertinggi",
    "Activity_status": "Taraf_aktiviti",
    "Occupation": "Pekerjaan",
    "Industry": "Industri",
}
df_p1 = df_p1.rename(columns=dict_col_harmonise)
cols_df_p1 = pd.Series(df_p1.columns)

# %%
# Recode dummies
# income-earners
df_p1["Penerima_pendapatan"] = df_p1["Penerima_pendapatan"].astype("int")
df_p1.loc[
    df_p1["Penerima_pendapatan"] == 2, "Penerima_pendapatan"
] = 0  # receives income
df_p2.loc[
    df_p2["Penerima_pendapatan"] == 2, "Penerima_pendapatan"
] = 0  # receives income
# citizenship
df_p2["Kewarganegaraan"] = df_p2["Kewarganegaraan"].astype("int")
df_p1["Kewarganegaraan"] = df_p1["Kewarganegaraan"].astype("int")
df_p2.loc[df_p2["Kewarganegaraan"] == 2, "Kewarganegaraan"] = 0  # malaysian
df_p1.loc[df_p1["Kewarganegaraan"] == 2, "Kewarganegaraan"] = 0  # malaysian
# gender
df_p2["Jantina"] = df_p2["Jantina"].astype("int")
df_p1["Jantina"] = df_p1["Jantina"].astype("int")
df_p2.loc[df_p2["Jantina"] == 2, "Jantina"] = 0  # male
df_p1.loc[df_p1["Jantina"] == 2, "Jantina"] = 0  # male
# strata
df_p1["Strata"] = df_p1["Strata"].astype("int")
df_p1.loc[df_p1["Strata"] == 2, "Strata"] = 0  # urban
df_p2.loc[df_p2["Strata"] == "Luar Bandar", "Strata"] = 0  # urban
df_p2.loc[df_p2["Strata"] == "Bandar", "Strata"] = 1  # urban

# %%
# Recode ethnicity
dict_ethnicity = {1: "bumiputera", 2: "chinese", 3: "indian", 4: "others"}
df_p1["Kaum"] = df_p1["Kaum"].replace(dict_ethnicity)
dict_ethnicity = {
    "Bumiputera": "bumiputera",
    "Cina": "chinese",
    "India": "indian",
    "Lain-lain": "others",
}
df_p2["Kaum"] = df_p2["Kaum"].replace(dict_ethnicity)

# %%
# Recode state
dict_state = {
    1: "johor",
    2: "kedah",
    3: "kelantan",
    4: "melaka",
    5: "negeri_sembilan",
    6: "pahang",
    7: "penang",
    8: "perak",
    9: "perlis",
    10: "selangor",
    11: "terengganu",
    12: "sabah",
    13: "sarawak",
    14: "kuala_lumpur",
    15: "labuan",
    16: "putrajaya",
}
df_p1["Negeri"] = df_p1["Negeri"].replace(dict_state)
dict_state = {
    "Johor": "johor",
    "Kedah": "kedah",
    "Kelantan": "kelantan",
    "Melaka": "melaka",
    "Negeri Sembilan": "negeri_sembilan",
    "Pahang": "pahang",
    "Pulau Pinang": "penang",
    "Perak": "perak",
    "Perlis": "perlis",
    "Selangor": "selangor",
    "Terengganu": "terengganu",
    "Sabah": "sabah",
    "Sarawak": "sarawak",
    "WP Kuala Lumpur": "kuala_lumpur",
    "WP Labuan": "labuan",
    "WP Putrajaya": "putrajaya",
}
df_p2["Negeri"] = df_p2["Negeri"].replace(dict_state)

# %%
# Recode housing status (not available, but not essential)
# dict_house_status = \
#     {
#         1: 'owned',
#         2: 'rented',
#         3: 'squatters_owned',
#         4: 'squatters_rented',
#         5: 'quarters',
#         6: 'others',
#     }
# df_p1['JMR'] = df_p1['JMR'].replace(dict_house_status)

# %%
# Recode marriage status
df_p2["Taraf_perkahwinan"] = df_p2["Taraf_perkahwinan"].astype('int')
df_p1["Taraf_perkahwinan"] = df_p1["Taraf_perkahwinan"].astype('int')
dict_marriage_status = {
    1: "never",
    2: "married",
    3: "widowed",
    4: "divorced",
    5: "separated",
}
df_p2["Taraf_perkahwinan"] = df_p2["Taraf_perkahwinan"].replace(dict_marriage_status)
df_p1["Taraf_perkahwinan"] = df_p1["Taraf_perkahwinan"].replace(dict_marriage_status)

# %%
# Recode education
dict_education = {
    1: "diploma",
    2: "cert",
    3: "stpm",
    4: "spm",
    5: "pmr",
    6: "no_cert",
    7: "no_cert",
}
df_p1["Sijil_tertinggi"] = df_p1["Sijil_tertinggi"].replace(dict_education)
dict_education = {
    "Ijazah dan ke atas": "diploma",
    "Diploma/Sijil": "cert",
    "STPM atau yang setaraf": "stpm",
    "SPM atau yang setaraf": "spm",
    "PT3/PMR/SRP atau yang setaraf": "pmr",
    "UPSR/UPSRA atau yang setaraf": "no_cert",
    "Tiada Sijil": "no_cert",
}
df_p2["Sijil_tertinggi"] = df_p2["Sijil_tertinggi"].replace(dict_education)

# %%
# Recode emp_status
dict_emp_status = {
    1: "employer",
    2: "gov_employee",
    3: "priv_employee",
    4: "self_employed",
    5: "self_employed",
    6: "unpaid_fam",
    7: "unemployed",
    8: "housespouse",
    9: "student",
    10: "pensioner",
    11: "pensioner",
    12: "others",
    13: "child_not_at_work",
    14: "child_not_at_work",
    15: "others",
}
df_p1["Taraf_aktiviti"] = df_p1["Taraf_aktiviti"].replace(dict_emp_status)
dict_emp_status = {
    "Majikan": "employer",
    "Pekerja Kerajaan": "gov_employee",
    "Pekerja Swasta": "priv_employee",
    "Bekerja Sendiri(Berdaftar)": "self_employed",
    "Bekerja Sendiri(Tidak Berdaftar)": "self_employed",
    "Pekerja Keluarga Tanpa Gaji": "unpaid_fam",  # doesn't exist for HH basis
    "Penganggur": "unemployed",
    "Suri rumah/menjaga rumah": "housespouse",
    "Pelajar": "student",
    "Pesara Kerajaan": "pensioner",
    "Pesara Swasta": "pensioner",
    "Warga Emas": "others",
    "Kanak-Kanak Tidak Bersekolah": "child_not_at_work",
    "Bayi": "child_not_at_work",
    "Lain-lain": "others",
}
df_p2["Taraf_aktiviti"] = df_p2["Taraf_aktiviti"].replace(dict_emp_status)

# %%
# Redefine occupation
dict_occ = {
    1: "manager",
    2: "professional",
    3: "technician",
    4: "clerical",
    5: "services",
    6: "agriculture",
    7: "craft",
    8: "plant_operator",
    9: "elementary",
    0: "others",
}
df_p1["Pekerjaan"] = df_p1["Pekerjaan"].replace(dict_occ)
dict_occ = {
    "Pengurus": "manager",
    "Profesional": "professional",
    "Juruteknik dan Profesional Bersekutu": "technician",
    "Pekerja Sokongan Perkeranian": "clerical",
    "Pekerja Perkhidmatan dan Jualan": "services",
    "Pekerja Kemahiran Pertanian, Perhutanan, Penternakan dan Perikanan": "agriculture",
    "Pekerja Kemahiran dan Pekerja Pertukangan Yang Berkaitan": "craft",
    "Operator Mesin dan Loji, Dan Pemasang": "plant_operator",
    "Pekerja Asas": "elementary",
    "TTDL": "others",
}
df_p2["Pekerjaan"] = df_p2["Pekerjaan"].replace(dict_occ)

# %%
# Redefine industries
dict_ind = {
    2: "agriculture",
    3: "mining",
    4: "manufacturing",
    5: "elec_gas_steam_aircond",
    6: "water_sewer_waste",
    7: "construction",
    8: "wholesale_retail",
    9: "transport_storage",
    10: "accom_fb",
    11: "info_comm",
    12: "finance_insurance",
    13: "real_estate",
    14: "professional",
    15: "admin_support",
    16: "public_admin",
    17: "education",
    18: "health_social",
    19: "arts",
    20: "other_services",
    21: "household",
    22: "extra_territorial",
    1: "others",
}
df_p1["Industri"] = df_p1["Industri"].replace(dict_ind)
dict_ind = {
    "Agriculture, Forestry & Fishing": "agriculture",
    "Mining & Quarrying": "mining",
    "Manufacturing": "manufacturing",
    "Electric,Gas,Steam & Air Conditioning Supply": "elec_gas_steam_aircond",
    "Water Supply; Sewerage, Waste Management & Remediation Activities": "water_sewer_waste",
    "Construction": "construction",
    "Wholesale & Retail Trade; Repair of Motor Vehicles & Motorcycles": "wholesale_retail",
    "Transportation & Storage": "transport_storage",
    "Accomodation & Food Service Activities": "accom_fb",
    "Information & Communication": "info_comm",
    "Financial & Insurance/Takaful Activities": "finance_insurance",
    "Real Estate Activities": "real_estate",
    "Professional, Scientific & Technical Activities": "professional",
    "Administrative & Support Service Activities": "admin_support",
    "Public Administration & Defence; Compulsory Social Activities": "public_admin",
    "Education": "education",
    "Human Health & Social Work Activities": "health_social",
    "Arts, Entertainment & Recreation": "arts",
    "Other Service Activities": "other_services",
    "Activities of Households As Employers; Undifferentiated Goods & Services - Producing Activities of Households For Own Us": "household",
    "Activities of Extraterritorial Organizations and Bodies": "extra_territorial",
    "Homemakers etc": "others",
}
df_p2["Industri"] = df_p2["Industri"].replace(dict_ind)

# %%
# p1: Separate column for number of income-generating members
p1_income_gen = (
    df_p1.groupby("HID")["Penerima_pendapatan"]
    .sum()
    .reset_index()
    .rename(columns={"Penerima_pendapatan": "income_gen_members"})
)

# %%
# p1: Non-income-generating adult females (18-59)
p1_idle_women = df_p1[
    (
        ((df_p1["Umur"] >= 18) & (df_p1["Umur"] < 60))
        & (df_p1["Penerima_pendapatan"] == 0)
        & (df_p1["Jantina"] == 0)
    )
].copy()  # keep only rows corresponding to adult females not working
p1_idle_women = (
    p1_idle_women.groupby("HID")["Penerima_pendapatan"]
    .count()
    .reset_index()
    .rename(columns={"Penerima_pendapatan": "non_working_adult_females"})
)

# %%
# p1: Income-generating adult females (18-59)
p1_working_women = df_p1[
    (
        ((df_p1["Umur"] >= 18) & (df_p1["Umur"] < 60))
        & (df_p1["Penerima_pendapatan"] == 1)
        & (df_p1["Jantina"] == 0)
    )
].copy()  # keep only rows corresponding to working adult females
p1_working_women = (
    p1_working_women.groupby("HID")["Penerima_pendapatan"]
    .count()
    .reset_index()
    .rename(columns={"Penerima_pendapatan": "working_adult_females"})
)

# %%
# p1: Separate column for <= 12 year-olds, <= 17 year-olds, and elderly (>= 60 year olds)
p1_kids = df_p1[["HID", "Umur"]].copy()
p1_kids.loc[p1_kids["Umur"] <= 17, "kid"] = 1
p1_kids.loc[p1_kids["Umur"] <= 12, "child"] = 1
p1_kids.loc[(p1_kids["Umur"] > 12) & (p1_kids["Umur"] <= 17), "adolescent"] = 1
p1_kids.loc[p1_kids["Umur"] >= 60, "elderly"] = 1
p1_kids.loc[(p1_kids["Umur"] >= 18) & (p1_kids["Umur"] <= 64), "working_age2"] = 1
p1_kids.loc[p1_kids["Umur"] >= 65, "elderly2"] = 1
p1_kids.loc[(p1_kids["Umur"] >= 18) & (p1_kids["Umur"] <= 59), "working_age"] = 1
for i in [
    "kid",
    "child",
    "adolescent",
    "elderly",
    "working_age2",
    "elderly2",
    "working_age",
]:
    p1_kids.loc[p1_kids[i].isna(), i] = 0
p1_kids = (
    p1_kids.groupby("HID")[
        [
            "kid",
            "child",
            "adolescent",
            "elderly",
            "working_age2",
            "elderly2",
            "working_age",
        ]
    ]
    .sum()
    .reset_index()
)

# %%
# p1: Keep only head of households
print(
    tabulate(
        pd.crosstab(df_p1["Penerima_pendapatan"], df_p1["Relationship"]),
        showindex=True,
        headers="keys",
        tablefmt="pretty",
    )
)
df_p1 = df_p1[df_p1["Relationship"] == "01"].copy()  # str

# %%
# inc: Compute gross income (ALL INCOME + GROSS TRANSFERS)
# df_inc['INCS07_hh'] = df_inc['INCS01_hh'] + df_inc['INCS02_hh'] + df_inc['INCS03_hh'] + df_inc['INCS05_hh']

# %%
# inc: Trim to retain only household income columns
cols_inc = ["INCS01_hh", "INCS02_hh", "INCS03_hh", "INCS05_hh", "INCS07_hh"]
df_inc = df_inc[["HID"] + cols_inc]  # already in annual basis

# %%
# p2: Rename expenditure columns
cols_exp = [i for i in df_p2.columns if i.startswith("g")]
cols_exp_revised = [i.replace("g", "cons_", 1) for i in cols_exp]
dict_exp_revised = dict(zip(cols_exp, cols_exp_revised))
df_p2 = df_p2.rename(columns=dict_exp_revised)

# %%
# p2: Calculate special expenditure categories
# fuel
df_p2["cons_0722_fuel"] = (
    df_p2["cons_072210"]
    + df_p2["cons_072220"]
    + df_p2["cons_072230"]
    + df_p2["cons_072240"]
)
# electricity
df_p2["cons_0451_elec"] = df_p2["cons_045100"].copy()
# transport ex cars, motorcycles, bicycles, and servicing
cols_cons_07_ex_bigticket = [
    i
    for i in df_p2.columns
    if not (
        (i.startswith("cons_0711"))
        or (i.startswith("cons_0712"))
        or (i.startswith("cons_0713"))
        or (i.startswith("cons_0723"))
    )
    and not (i == "cons_07")
    and (i.startswith("cons_07"))
    and ("cons_" in i)
]
df_p2["cons_07_ex_bigticket"] = df_p2[cols_cons_07_ex_bigticket].sum(
    axis=1
)  # column sums

# %%
# p2: Trim spending categories
cols_exp_2d = ["cons_0" + str(i) for i in range(1, 10)] + [
    "cons_1" + str(i) for i in range(0, 4)
]
cols_special = ["cons_0722_fuel", "cons_0451_elec", "cons_07_ex_bigticket"]
cols_exp_drop = [
    i for i in df_p2.columns if not (i in cols_exp_2d + cols_special) and ("cons_" in i)
]
for col in cols_exp_drop:
    del df_p2[col]

# %%
# p1 + p2
col_overlap = [i for i in df_p2.columns if i in df_p1.columns and ("HID" not in i)]
df = df_p2.merge(df_p1, on="HID", how="left", validate="one_to_one")
for i in tqdm(col_overlap):
    df.loc[df[i + "_x"].isna(), i + "_x"] = df[i + "_y"]  # same as combine_first
    del df[i + "_y"]  # left (block 1; hh basis) is dominant
    df = df.rename(columns={i + "_x": i})
del df_p2
del df_p1

# %%
# p1 + p2 + inc
df = df.merge(df_inc, on="HID", how="left", validate="one_to_one")

# %%
# p1 + p2 + inc + income-gen
df = df.merge(p1_income_gen, on="HID", how="left", validate="one_to_one")
del p1_income_gen
df.loc[df["income_gen_members"].isna(), "income_gen_members"] = 0

# %%
# p1 + p2 + inc + kids
df = df.merge(p1_kids, on="HID", how="left", validate="one_to_one")
del p1_kids
for i in ["child", "adolescent", "elderly"]:
    df.loc[df[i].isna(), i] = 0

# %%
# p1 + p2 + inc + idle adult women
df = df.merge(p1_idle_women, on="HID", how="left", validate="one_to_one")
del p1_idle_women
df.loc[df["non_working_adult_females"].isna(), "non_working_adult_females"] = 0

# %%
# p1 + p2 + inc + working adult women
df = df.merge(p1_working_women, on="HID", how="left", validate="one_to_one")
del p1_working_women
df.loc[df["working_adult_females"].isna(), "working_adult_females"] = 0

# %%
# p1 + p2 + inc: mismatched labels

# %%
# p1 + p2 + inc: redundant columns
for col in [
    "NGDP",
    "Relationship",
    "Perbelanjaan_bulanan",
    "Pendapatan_bulanan",
    "Penerima_pendapatan",
]:
    del df[col]

# %%
# Compute total spending
cols_cons_01_12 = ["cons_0" + str(i) for i in range(1, 10)] + [
    "cons_1" + str(i) for i in range(0, 3)
]
df["cons_01_12"] = df[cols_cons_01_12].sum(axis=1)
cols_cons_01_13 = ["cons_0" + str(i) for i in range(1, 10)] + [
    "cons_1" + str(i) for i in range(0, 4)
]
df["cons_01_13"] = df[cols_cons_01_13].sum(axis=1)

# %%
# Margins
df["gross_margin"] = (df["INCS07_hh"] / 12) - df["cons_01_12"]
# df['net_margin'] = (df['INCS08_hh'] / 12) - df['Jumlah_perbelanjaan_01_12_sebula']

# %%
# Rename columns
dict_rename = {
    "HID": "id",
    "NOIR": "member_no",
    "Wajaran": "svy_weight",
    "Saiz_isirumah": "hh_size",
    # 'income_gen_members': 'income_gen_members',
    # 'working_adult_females': '',
    # 'non_working_adult_females': '',
    "Kaum": "ethnicity",
    "Negeri": "state",
    "Strata": "urban",
    "INCS01_hh": "salaried_wages",
    "INCS02_hh": "other_wages",
    "INCS03_hh": "asset_income",
    "INCS05_hh": "gross_transfers",
    "INCS06_hh": "net_transfers",
    "INCS07_hh": "gross_income",
    # 'INCS08_hh': 'net_income',
    # 'gross_margin': 'gross_margin',
    # 'net_margin': 'net_margin',
    # 'JMR': 'house_status',
    # 'Jumlah_perbelanjaan_01_12_sebula': 'cons_01_12',
    # 'Jumlah_perbelanjaan_01_13_sebula': 'cons_01_13',
    # 'Jumlah_pendapatan_sebulan': 'monthly_income',
    # 'Relationship': 'member_relation',
    "Jantina": "male",
    "Umur": "age",
    "Kewarganegaraan": "malaysian",
    "Taraf_perkahwinan": "marriage",
    # 'Penerima_pendapatan': 'receives_income',
    "Taraf_aktiviti": "emp_status",
    "Industri": "industry",
    "Pekerjaan": "occupation",
    "Sijil_tertinggi": "education",
    # 'kid': '',
    # 'child': '',
    # 'adolescent': '',
    # 'elderly': '',
    # 'working_age2': '',
    # 'working_age': '',
    # 'elderly2': '',
    # 'cons_01': '',
    # 'cons_02': '',
    # 'cons_03': '',
    # 'cons_04': '',
    # 'cons_05': '',
    # 'cons_06': '',
    # 'cons_07': '',
    # 'cons_08': '',
    # 'cons_09': '',
    # 'cons_10': '',
    # 'cons_11': '',
    # 'cons_12': '',
    # 'cons_13': '',
    # 'cons_0722_fuel': '',
    # 'cons_07_ex_bigticket': '',
    # 'cons_0451_elec': ''
}
df = df.rename(columns=dict_rename)

# %%
# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)):
    raise NotImplementedError("IDs are not unique")
df["id"] = df.reset_index().index
df = df.reset_index(drop=True)

# %%
# Post-merge: convert income, expenditure, and margins into monthly per capita
for i in [
    "salaried_wages",
    "other_wages",
    "asset_income",
    "gross_transfers",
    "gross_income",
]:
    # df[i] = df[i] / 12  # Convert from annual to monthly
    df[i] = df[i] / (df['hh_size'] * 12)
for i in ['cons_01_12', 'cons_01_13'] + \
         ['cons_0' + str(i) for i in range(1, 10)] + \
         ['cons_' + str(i) for i in range(11, 14)] + \
         ['cons_0722_fuel', 'cons_0451_elec', 'cons_07_ex_bigticket']:
    df[i] = df[i] / df['hh_size']
for i in ['gross_margin']:
    df[i] = df[i] / df['hh_size']

# %%
# Post-merge: birth year
df["birth_year"] = 2022 - df["age"]

# %%
# Drop NAs
df = df.dropna(
    axis=0, how="any"
)  # drop rows (IMPORTANT: HIES SOMEHOW HAS NO INCOME INFO FOR ALL FOREIGNERS)

# %%
# Harmonise dtypes
dict_dtypes_22 = {
    "state": "str",
    "education": "str",
    "ethnicity": "str",
    "malaysian": "int",
    "income_gen_members": "int",
    "working_adult_females": "int",
    "non_working_adult_females": "int",
    "kid": "int",
    "adolescent": "int",
    "child": "int",
    "elderly": "int",
    "working_age2": "int",
    "working_age": "int",
    "elderly2": "int",
    "male": "int",
    "marriage": "str",
    "emp_status": "str",
    "industry": "str",
    "occupation": "str",
    "hh_size": "int",
    "urban": "int",
    "salaried_wages": "float",
    "other_wages": "float",
    "asset_income": "float",
    "gross_transfers": "float",
    # 'net_transfers': 'float',
    # 'net_income': 'float',
    "cons_01_12": "float",
    "cons_01_13": "float",
    # 'monthly_income': 'float',
    "gross_income": "float",
    "age": "int",
    "birth_year": "int",
    "cons_01": "float",
    "cons_02": "float",
    "cons_03": "float",
    "cons_04": "float",
    "cons_05": "float",
    "cons_06": "float",
    "cons_07": "float",
    "cons_08": "float",
    "cons_09": "float",
    "cons_10": "float",
    "cons_11": "float",
    "cons_12": "float",
    "cons_13": "float",
    "cons_0722_fuel": "float",
    "cons_07_ex_bigticket": "float",
    "cons_0451_elec": "float",
    "gross_margin": "float",
    # 'net_margin': 'float',
}
df = df.astype(dict_dtypes_22)

# %%
# Inspect categorical values
cols_cat_inspect = [
    "male",
    "malaysian",
    "marriage",
    "hh_size",
    "ethnicity",
    "emp_status",
    "occupation",
    "industry",
    "education",
    "state",
    "urban",
    "income_gen_members",
    "kid",
    "child",
    "adolescent",
    "elderly",
    "working_age2",
    "elderly2",
    "working_age",
    "non_working_adult_females",
    "working_adult_females",
]
for col in cols_cat_inspect:
    print(col + ":" + ", ".join(list(df[col].astype("str").unique())))

# %%
# V --- Save output
df.to_parquet(path_data + "hies_2022_consol.parquet", compression="brotli")

# %%
# X --- Notify
telsendmsg(
    conf=tel_config, msg="impact-household --- process_raw_2022: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
