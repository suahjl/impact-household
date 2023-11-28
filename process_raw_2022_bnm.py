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
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/hies_2022_bnm/'

# %%
# I --- Load b1
df_b1 = pd.read_spss(path_data + "HES 2022 Blok 1-isirumah.sav")
df_b1

# %%
# II --- Load b2
df_b2 = pd.read_spss(path_data + "HES 2022 Blok 2-ahli.sav")
df_b2

# %%
# III --- Load b3
df_b3 = pd.read_spss(path_data + "HES 2022 Blok 3-perbelanjaan01_13.sav")
df_b3

# %%
# IV --- Clean & Consolidate

# %%
# b2: Recode dummies
df_b2['Penerima_Pendapatan'] = df_b2['Penerima_Pendapatan'].astype("int")
df_b2.loc[df_b2['Penerima_Pendapatan'] == 2, 'Penerima_Pendapatan'] = 0  # receives income

df_b2['Citizenship'] = df_b2['Citizenship'].astype("int")
df_b2.loc[df_b2['Citizenship'] == 2, 'Citizenship'] = 0  # malaysian

df_b2['Jantina'] = df_b2['Jantina'].astype("int")
df_b2.loc[df_b2['Jantina'] == 2, 'Jantina'] = 0  # male

df_b1['Strata'] = df_b1['Strata'].astype("str")
df_b1.loc[df_b1['Strata'] == "Rural", 'Strata'] = 0  # urban
df_b1.loc[df_b1['Strata'] == "Urban", 'Strata'] = 1  # urban

df_b2['Strata'] = df_b2['Strata'].astype("str")
df_b2.loc[df_b2['Strata'] == "Luar bandar", 'Strata'] = 0  # urban
df_b2.loc[df_b2['Strata'] == "Bandar", 'Strata'] = 1  # urban

# %%
# b1 & b2: Recode ethnicity
df_b1['Ethnic'] = df_b1['Ethnic'].astype("str")
df_b2['Ethnic'] = df_b2['Ethnic'].astype("str")
dict_ethnicity = \
    {
        "Bumiputera": 'bumiputera',
        "Chinese": 'chinese',
        "Cina": 'chinese',
        "Indian": 'indian',
        "India": 'indian',
        "Others": 'others',
        "Lain-lain": 'others'
    }
df_b1['Ethnic'] = df_b1['Ethnic'].replace(dict_ethnicity)
df_b2['Ethnic'] = df_b2['Ethnic'].replace(dict_ethnicity)

# %%
# b1: Recode state
df_b1["State"] = df_b1["State"].astype("int")
dict_state = \
    {
        1: 'johor',
        2: 'kedah',
        3: 'kelantan',
        4: 'melaka',
        5: 'negeri_sembilan',
        6: 'pahang',
        7: 'penang',
        8: 'perak',
        9: 'perlis',
        10: 'selangor',
        11: 'terengganu',
        12: 'sabah',
        13: 'sarawak',
        14: 'kuala_lumpur',
        15: 'labuan',
        16: 'putrajaya',
    }
df_b1['State'] = df_b1['State'].replace(dict_state)

# %%
# b1: Recode housing status
# dict_house_status = \
#     {
#         1: 'owned',
#         2: 'rented',
#         3: 'squatters_owned',
#         4: 'squatters_rented',
#         5: 'quarters',
#         6: 'others',
#     }
# df_b1['JMR'] = df_b1['JMR'].replace(dict_house_status)

# %%
# b2: Recode marriage status
df_b2['Taraf_perkahwinan'] = df_b2['Taraf_perkahwinan'].astype('int')
dict_marriage_status = \
    {
        1: 'never',
        2: 'married',
        3: 'widowed',
        4: 'divorced',
        5: 'separated',
    }
df_b2['Taraf_perkahwinan'] = df_b2['Taraf_perkahwinan'].replace(dict_marriage_status)

# %%
# b2: Recode education
df_b2['Sijil_tertinggi_diperoleh'] = df_b2['Sijil_tertinggi_diperoleh'].astype('str')
dict_education = \
    {
        'Ijazah/Diploma Lanjutan': 'diploma',
        'Diploma/Sijil': 'cert',
        'STPM': 'stpm',
        'SPM/ SPMV': 'spm',
        'PT3/PMR/SRP': 'pmr',
        'Tiada sijil': 'no_cert',
        'UPSR/UPSRA': 'no_cert'
    }
df_b2['Sijil_tertinggi_diperoleh'] = df_b2['Sijil_tertinggi_diperoleh'].replace(dict_education)

# %%
# b2: Recode emp_status
df_b2['Taraf_aktiviti'] = df_b2['Taraf_aktiviti'].astype('str')
dict_emp_status = \
    {
        'Majikan': 'employer',
        'Pekerja kerajaan': 'gov_employee',
        'Pekerja swasta': 'priv_employee',
        'Pekerja sendiri (tidak berdaftar)': 'self_employed',
        'Pekerja sendiri (berdaftar)': 'self_employed',
        'Pekerja keluarga tanpa gaji': 'unpaid_fam',
        'Penganggur': 'unemployed',
        'Suri rumah/ menjaga rumah': 'housespouse',
        'Pelajar': 'student',
        'Pesara kerajaan': 'pensioner',
        'Pesara swasta': 'pensioner',
        'Warga emas': 'others',
        'Kanak-kanak tidak bersekolah': 'child_not_at_work',
        'Bayi': 'child_not_at_work',
        'Lain-lain': 'others',
    }
df_b2['Taraf_aktiviti'] = df_b2['Taraf_aktiviti'].replace(dict_emp_status)

# %%
# b2: redefine occupation
df_b2['Pekerjaan'] = df_b2['Pekerjaan'].astype("str")
dict_occ = \
    {
        "Pengurus": 'manager',
        "Profesional": 'professional',
        "Juruteknik dan Profesional Bersekutu": 'technician',
        "Pekerja Sokongan Perkeranian": 'clerical',
        "Pekerja Perkhidmatan dan Jualan": 'services',
        "Pekerja Mahir Pertanian, Perhutanan, Penternakan, dan Perikanan": 'agriculture',
        "Pekerja Kemahiran dan Pekerja Pertukangan yang Berkaitan": 'craft',
        "Operator Mesin dan Loji, dan Pemasangan": 'plant_operator',
        "Pekerja Asas": 'elementary',
        "Pekerjaan yang Tidak dikelaskan dimana-mana": 'others',
    }
df_b2['Pekerjaan'] = df_b2['Pekerjaan'].replace(dict_occ)

# %%
# b2: redefine industries
df_b2['Industri'] = df_b2['Industri'].astype("str")
dict_ind = \
    {
        "Pertanian, Perhutanan & Perikanan;": 'agriculture',
        "Perlombongan & Pengkuarian": 'mining',
        "Pembuatan": 'manufacturing',
        "Bekalan Elektrik, Gas, Wap & Pendingin Udara": 'elec_gas_steam_aircond',
        "Bekalan Air;Pembentungan, Pengurusan Sisa & Aktiviti Pemulih": 'water_sewer_waste',
        "Pembinaan": 'construction',
        "Perdagangan Borong & Runcit;Pembaikan Kenderaan Bermotor & Motosikal": 'wholesale_retail',
        "Pengangkutan & Penyimpanan": 'transport_storage',
        "Penginapan & Aktiviti Perkhidmatan Makanan & Minuman": 'accom_fb',
        "Maklumat & Komunikasi": 'info_comm',
        "Aktiviti Kewangan & Insurans/Takaful": 'finance_insurance',
        "Aktiviti Hartanah": 'real_estate',
        "Aktiviti Professional, Saintifik & TeknikalAktiviti Professional, Saintifik & Teknikal": 'professional',
        "Aktiviti Pentadbiran & Khidmat Sokongan": 'admin_support',
        "Pentadbiran Awam & Pertahanan;Aktiviti Keselamatan Sosial Wajib": 'public_admin',
        "Pendidikan": 'education',
        "Aktiviti Kesihatan Kemanusiaan & Kerja Sosial": 'health_social',
        "Kesenian, Hiburan & Rekreasi": 'arts',
        'Aktiviti Perkhimatan Lain': 'other_services',
        "Aktiviti Isi Rumah Sebagai Majikan, Aktiviti Mengeluarkan Barangan dan Perkhidmatan yang tidak dapat dibezakan": 'household',
        "Organisasi dan badan di luar wilayah": 'extra_territorial',
        "Industri tidak dikelaskan di mana-mana": 'others'
    }
df_b2['Industri'] = df_b2['Industri'].replace(dict_ind)

# %%
# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('HID')['Penerima_Pendapatan'].sum().reset_index().rename(columns={'Penerima_Pendapatan': 'income_gen_members'})

# %%
# b2: non-income-generating adult females (18-59)
b2_idle_women = df_b2[(((df_b2['Umur'] >= 18) & (df_b2['Umur'] < 60)) &
                       (df_b2['Penerima_Pendapatan'] == 0) & (df_b2[
                                                 'Jantina'] == 0))].copy()  # keep only rows corresponding to adult females not working
b2_idle_women = b2_idle_women.groupby('HID')['Penerima_Pendapatan'] \
    .count() \
    .reset_index() \
    .rename(columns={'Penerima_Pendapatan': 'non_working_adult_females'})

# %%
# b2: Income-generating adult females (18-59)
b2_working_women = df_b2[(((df_b2['Umur'] >= 18) & (df_b2['Umur'] < 60)) &
                          (df_b2['Penerima_Pendapatan'] == 1) & (
                                      df_b2['Jantina'] == 0))].copy()  # keep only rows corresponding to working adult females
b2_working_women = b2_working_women.groupby('HID')['Penerima_Pendapatan'] \
    .count() \
    .reset_index() \
    .rename(columns={'Penerima_Pendapatan': 'working_adult_females'})

# %%
# b2: Separate column for <= 12 year-olds, <= 17 year-olds, and elderly (>= 60 year olds)
b2_kids = df_b2[['HID', 'Umur']].copy()
b2_kids.loc[b2_kids['Umur'] <= 17, 'kid'] = 1
b2_kids.loc[b2_kids['Umur'] <= 12, 'child'] = 1
b2_kids.loc[(b2_kids['Umur'] > 12) & (b2_kids['Umur'] <= 17), 'adolescent'] = 1
b2_kids.loc[b2_kids['Umur'] >= 60, 'elderly'] = 1
b2_kids.loc[(b2_kids['Umur'] >= 18) & (b2_kids['Umur'] <= 64), 'working_age2'] = 1
b2_kids.loc[b2_kids['Umur'] >= 65, 'elderly2'] = 1
b2_kids.loc[(b2_kids['Umur'] >= 18) & (b2_kids['Umur'] <= 59), 'working_age'] = 1
for i in ['kid', 'child', 'adolescent', 'elderly', 'working_age2', 'elderly2', 'working_age']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('HID')[
    ['kid', 'child', 'adolescent', 'elderly', 'working_age2', 'elderly2', 'working_age']].sum().reset_index()

# %%
# b2: Keep only head of households
df_b2['Perhubungan'] = df_b2['Perhubungan'].astype("int")
print(tabulate(pd.crosstab(df_b2['Penerima_Pendapatan'], df_b2['Perhubungan']), showindex=True, headers='keys', tablefmt="pretty"))
df_b2 = df_b2[df_b2['Perhubungan'] == 1].copy()

# %%
# b1: Compute gross income (ALL INCOME + GROSS TRANSFERS)
df_b1['INCS07_hh'] = df_b1['INCS01_hh'] + df_b1['INCS02_hh'] + df_b1['INCS03_hh'] + df_b1['INCS05_hh']
del df_b1['ginc_hh']

# %%
# b1 + b2
col_overlap = [i for i in df_b1.columns if i in df_b2.columns and ('HID' not in i)]
df = df_b1.merge(df_b2, on='HID', how='left', validate='one_to_one')
for i in tqdm(col_overlap):
    df.loc[df[i + '_x'].isna(), i + '_x'] = df[i + '_y']  # same as combine_first
    del df[i + '_y']  # left (block 1) is dominant
    df = df.rename(columns={i + '_x': i})
del df_b1
del df_b2

# %%
# b1 + b2 + income-gen
df = df.merge(b2_income_gen, on='HID', how='left', validate='one_to_one')
del b2_income_gen
df.loc[df['income_gen_members'].isna(), 'income_gen_members'] = 0

# %%
# b1 + b2 + kids
df = df.merge(b2_kids, on='HID', how='left', validate='one_to_one')
del b2_kids
for i in ['child', 'adolescent', 'elderly']:
    df.loc[df[i].isna(), i] = 0

# %%
# b1 + b2 + idle adult women
df = df.merge(b2_idle_women, on='HID', how='left', validate='one_to_one')
del b2_idle_women
df.loc[df['non_working_adult_females'].isna(), 'non_working_adult_females'] = 0

# %%
# b1 + b2 + working adult women
df = df.merge(b2_working_women, on='HID', how='left', validate='one_to_one')
del b2_working_women
df.loc[df['working_adult_females'].isna(), 'working_adult_females'] = 0


# %%
# b1 + b2: redundant columns
for i in ['NOIR', 'Perhubungan', 'Penerima_Pendapatan', 'Negeri']:
    del df[i]

# %%
# b3: change to int format
for col in ["TwoD", "FourD", "SixD"]:
    df_b3[col] = df_b3[col].astype("int")

# %%
# b3: isolate special subitems
# Fuel and electricity
cons_fuel = df_b3[df_b3['FourD'].isin([722, 451])]
cons_fuel = cons_fuel.groupby(['HID', 'FourD'])['amaun'].sum().reset_index()
cons_fuel = pd.pivot(cons_fuel, index='HID', columns='FourD', values='amaun').reset_index()  # long to wide
cons_fuel = cons_fuel.rename(
    columns={
        722: 'cons_0722_fuel',
        451: 'cons_0451_elec'
    }
)
# Transport ex cars, motorcycles, bicycles, and servicing
cons_transport_ex_bigticket = \
    df_b3[~(df_b3['FourD'].isin(
        [711, 712, 713, 723]
    )) & (df_b3['TwoD'] == 7)]  # exclude cars, motorcycles, bicycles, and servicing
cons_transport_ex_bigticket = cons_transport_ex_bigticket.groupby(['HID', 'TwoD'])['amaun'].sum().reset_index()
cons_transport_ex_bigticket = \
    pd.pivot(cons_transport_ex_bigticket, index='HID', columns='TwoD', values='amaun').reset_index()  # long to wide
cons_transport_ex_bigticket = cons_transport_ex_bigticket.rename(
    columns={
        7: 'cons_07_ex_bigticket'
    }
)

# %%
# b3: group expenditure by items
cons_two_digits = df_b3.groupby(['HID', 'TwoD'])['amaun'].sum().reset_index()
cons_two_digits = pd.pivot(cons_two_digits, index='HID', columns='TwoD', values='amaun').reset_index()  # long to wide
cons_two_digits = cons_two_digits.rename(
    columns={
        1: 'cons_01',
        2: 'cons_02',
        3: 'cons_03',
        4: 'cons_04',
        5: 'cons_05',
        6: 'cons_06',
        7: 'cons_07',
        8: 'cons_08',
        9: 'cons_09',
        10: 'cons_10',
        11: 'cons_11',
        12: 'cons_12',
        13: 'cons_13',
    }
)
cons_two_digits = cons_two_digits.fillna(0)

# %%
# b1 + b2 + b3
df = df.merge(cons_two_digits, on='HID', how='left', validate='one_to_one')
df = df.merge(cons_fuel, on='HID', how='left', validate='one_to_one')
df = df.merge(cons_transport_ex_bigticket, on='HID', how='left', validate='one_to_one')
del df_b3
del cons_two_digits
del cons_fuel
del cons_transport_ex_bigticket

# %%
# Compute aggregate consumption
cols_cons_01_12 = ['cons_0' + str(i) for i in range(1, 10)] + ['cons_' + str(i) for i in range(11, 13)]
cols_cons_01_13 = ['cons_0' + str(i) for i in range(1, 10)] + ['cons_' + str(i) for i in range(11, 14)]
df["cons_01_12"] = df[cols_cons_01_12].sum(axis=1)
# df["cons_01_13"] = df[cols_cons_01_13].sum(axis=1)
del df["jum_g01g13"]
# del df["jum_g01g90"]
df = df.rename(columns={"jum_g01g90": "cons_01_13"})

# %%
# Margins
df['gross_margin'] = (df['INCS07_hh'] / 12) - df['cons_01_12']
# df['net_margin'] = (df['INCS08_hh'] / 12) - df['cons_01_13']

# %%
# Rename columns to be more intuitive
dict_rename = \
    {
        'HID': 'id',
        # 'NOIR': 'member_no',
        'wgt_hes': 'svy_weight',
        'HH_saiz': 'hh_size',
        # 'income_gen_members': 'income_gen_members',
        # 'working_adult_females': '',
        # 'non_working_adult_females': '',
        'Ethnic': 'ethnicity',
        'State': 'state',
        'Strata': 'urban',
        'INCS01_hh': 'salaried_wages',
        'INCS02_hh': 'other_wages',
        'INCS03_hh': 'asset_income',
        'INCS41_hh': 'remittance',
        'INCS42_hh': 'alimony',
        'INCS43_hh': 'bursary',
        'INCS44_hh': 'pension',
        'INCS45_hh': 'periodic',
        'INCS46_hh': 'gifts',
        'INCS05_hh': 'gross_transfers',
        'INCS06_hh': 'net_transfers',
        'INCS07_hh': 'gross_income',
        # 'INCS08_hh': 'net_income',
        # 'gross_margin': 'gross_margin',
        # 'net_margin': 'net_margin',
        # 'JMR': 'house_status',
        # 'cons_01_12': 'cons_01_12',
        # 'cons_01_13': 'cons_01_13',
        'INCS07_hh_m': 'monthly_income',
        # 'Perhubungan': 'member_relation',
        'Jantina': 'male',
        'Umur': 'age',
        'Citizenship': 'malaysian',
        'Taraf_perkahwinan': 'marriage',
        # 'Penerima_Pendapatan': 'receives_income',
        'Taraf_aktiviti': 'emp_status',
        'Industri': 'industry',
        'Pekerjaan': 'occupation',
        'Sijil_tertinggi_diperoleh': 'education',
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
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# %%
# post-merge: convert income, expenditure, and margins into monthly per capita
for i in ['salaried_wages', 'other_wages', 'asset_income',
          'gross_transfers', 'net_transfers', 'gross_income']:
    # df[i] = df[i] / (12)
    df[i] = df[i] / (df['hh_size'] * 12)
for i in ['cons_01_12', 'cons_01_13'] + \
         ['cons_0' + str(i) for i in range(1, 10)] + \
         ['cons_' + str(i) for i in range(11, 14)] + \
         ['cons_0722_fuel', 'cons_0451_elec', 'cons_07_ex_bigticket']:
    df[i] = df[i] / df['hh_size']
for i in ['gross_margin']:
    df[i] = df[i] / df['hh_size']

# %%
# post-merge: birth year
df['birth_year'] = 2022 - df['age']  # check year

# %%
# Drop NAs
df = df.dropna(axis=0, how='any')

# %%
# Harmonise dtypes
dict_dtypes_22 = \
    {
        'state': 'str',
        'education': 'str',
        'ethnicity': 'str',
        'malaysian': 'int',
        'income_gen_members': 'int',
        'working_adult_females': 'int',
        'non_working_adult_females': 'int',
        'kid': 'int',
        'adolescent': 'int',
        'child': 'int',
        'elderly': 'int',
        'working_age2': 'int',
        'working_age': 'int',
        'elderly2': 'int',
        'male': 'int',
        'marriage': 'str',
        'emp_status': 'str',
        'industry': 'str',
        'occupation': 'str',
        'hh_size': 'int',
        'urban': 'int',
        'salaried_wages': 'float',
        'other_wages': 'float',
        'asset_income': 'float',
        'gross_transfers': 'float',
        'net_transfers': 'float',
        # 'net_income': 'float',
        'cons_01_12': 'float',
        'cons_01_13': 'float',
        # 'monthly_income': 'float',
        'gross_income': 'float',
        'age': 'int',
        'birth_year': 'int',
        'cons_01': 'float',
        'cons_02': 'float',
        'cons_03': 'float',
        'cons_04': 'float',
        'cons_05': 'float',
        'cons_06': 'float',
        'cons_07': 'float',
        'cons_08': 'float',
        'cons_09': 'float',
        'cons_10': 'float',
        'cons_11': 'float',
        'cons_12': 'float',
        'cons_13': 'float',
        'cons_0722_fuel': 'float',
        'cons_0451_elec': 'float',
        'cons_07_ex_bigticket': 'float',
        'gross_margin': 'float',
        # 'net_margin': 'float',
    }
df = df.astype(dict_dtypes_22)

# %%
# V --- Save output
df.to_parquet(path_data + 'hies_2022_consol.parquet', compression='brotli')

# %%
# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2022_bnm: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')

# %%
