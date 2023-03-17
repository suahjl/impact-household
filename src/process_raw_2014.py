import pandas as pd
from src.helper import telsendmsg, telsendimg, telsendfiles
from tabulate import tabulate
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/hies_2014/'

# I --- Load b1
df_b1 = pd.read_csv(path_data + 'Blok1.txt', sep=',')
# df_b1.to_parquet(path_data + 'blok1.parquet', compression='brotli')

# II --- Load b2
df_b2 = pd.read_csv(path_data + 'Blok2.txt', sep=',')
# df_b2.to_parquet(path_data + 'blok2.parquet', compression='brotli')

# III --- Load b3
df_b3 = pd.read_csv(path_data + 'Blok3.txt', sep=',')
df_b3_item13 = pd.read_excel(path_data + 'HIES2014_Exp13.xlsx', sheet_name='Sheet1')
# df_b3.to_parquet(path_data + 'blok3.parquet', compression='brotli')

# IV --- Clean & Consolidate

# b2: Remove empty IDs
df_b2 = df_b2[~(df_b2['ID'].isna())].copy()

# b2: Recode dummies
df_b2.loc[df_b2['Penerima_Pendapatan'] == 2, 'Penerima_Pendapatan'] = 0  # receives income

df_b2.loc[df_b2['Kewarganegaraan'] == 'Warganegara', 'Kewarganegaraan'] = 1  # malaysian
df_b2.loc[df_b2['Kewarganegaraan'] == 'Bukan Warganegara', 'Kewarganegaraan'] = 0  # non-malaysian
df_b2['Kewarganegaraan'] = df_b2['Kewarganegaraan'].astype('int')

df_b2.loc[df_b2['Jantina'] == 2, 'Jantina'] = 0  # male

df_b1.loc[df_b1['Strata'] == 'Bandar', 'Strata'] = 1  # urban v rural
df_b1.loc[df_b1['Strata'] == 'Luar Bandar', 'Strata'] = 0  # urban v rural

# b1 & b2: Recode ethnicity
dict_ethnicity = \
    {
        'Bumiputera': 'bumiputera',
        'Bukan Bumiputera': 'non_bumiputera',
    }
df_b1['Kump_Etnik'] = df_b1['Kump_Etnik'].replace(dict_ethnicity)
df_b2['Kump_Etnik'] = df_b2['Kump_Etnik'].replace(dict_ethnicity)

# b1: Recode state
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
df_b1['Negeri'] = df_b1['Negeri'].replace(dict_state)

# b1: Recode housing status

# b2: Recode marriage status
dict_marriage_status = \
    {
        1: 'never',
        2: 'married',
        3: 'widowed',
        4: 'divorced',
        5: 'separated',
        9: 'never',  # if no info, assumed to be never married
    }
df_b2['Taraf_Perkahwinan'] = df_b2['Taraf_Perkahwinan'].replace(dict_marriage_status)

# b2: Recode education (basic)
dict_education_basic = \
    {
        'Menengah': 'secondary',
        'Rendah': 'primary',
        'Tertiari': 'tertiary',
        'Tiada Pendidikan Rasmi': 'no_educ',
    }
df_b2['Pencapaian_Pendidikan'] = df_b2['Pencapaian_Pendidikan'].replace(dict_education_basic)

# b2: Recode education
dict_education = \
    {
        1: 'diploma',
        2: 'cert',
        3: 'stpm',
        4: 'spm',
        5: 'pmr',
        6: 'no_cert',
    }
df_b2['Sijil_Tertinggi'] = df_b2['Sijil_Tertinggi'].replace(dict_education)

# b2: Recode emp_status
dict_emp_status = \
    {
        1: 'employer',
        2: 'gov_employee',
        3: 'priv_employee',
        4: 'self_employed',
        5: 'unpaid_fam',
        6: 'unemployed',
        7: 'housespouse',
        8: 'student',
        9: 'pensioner',
        10: 'others',
        11: 'child_not_at_work',
    }
df_b2['Taraf_Aktiviti'] = df_b2['Taraf_Aktiviti'].replace(dict_emp_status)

# b2: redefine occupation
dict_occ = \
    {
        1: 'manager',
        2: 'professional',
        3: 'technician',
        4: 'clerical',
        5: 'services',
        6: 'agriculture',
        7: 'craft',
        8: 'plant_operator',
        9: 'elementary',
        10: 'others',
        11: 'others',
    }
df_b2['Pekerjaan'] = df_b2['Pekerjaan'].replace(dict_occ)

# b2: redefine industries
dict_ind = \
    {
        1: 'agriculture',
        2: 'mining',
        3: 'manufacturing',
        4: 'elec_gas_steam_aircond',
        5: 'water_sewer_waste',
        6: 'construction',
        7: 'wholesale_retail',
        8: 'transport_storage',
        9: 'accom_fb',
        10: 'info_comm',
        11: 'finance_insurance',
        12: 'real_estate',
        13: 'professional',
        14: 'admin_support',
        15: 'public_admin',
        16: 'education',
        17: 'health_social',
        18: 'arts',
        19: 'other_services',
        20: 'household',
        21: 'extra_territorial',
        # 22: 'others'
    }
df_b2['Industri'] = df_b2['Industri'].replace(dict_ind)

# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('ID')['Penerima_Pendapatan'] \
    .sum() \
    .reset_index() \
    .rename(columns={'Penerima_Pendapatan': 'income_gen_members'})

# b2: Separate column for <= 12 year-olds, and <= 17 year-olds
b2_kids = df_b2[['ID', 'Umur']].copy()
b2_kids.loc[b2_kids['Umur'] <= 12, 'child'] = 1
b2_kids.loc[(b2_kids['Umur'] > 12) & (b2_kids['Umur'] <= 17), 'adolescent'] = 1
for i in ['child', 'adolescent']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('ID')[['child', 'adolescent']].sum().reset_index()

# b2: Keep only head of households
print(tabulate(pd.crosstab(df_b2['Penerima_Pendapatan'], df_b2['Perhubungan_KIR']), showindex=True, headers='keys',
               tablefmt="pretty"))
df_b2 = df_b2[df_b2['Perhubungan_KIR'] == 1].copy()

# b1 + b2
col_overlap = [i for i in df_b1.columns if i in df_b2.columns and ('ID' not in i)]
df = df_b1.merge(df_b2, on='ID', how='left', validate='one_to_one')
for i in tqdm(col_overlap):
    df.loc[df[i + '_x'].isna(), i + '_x'] = df[i + '_y']  # same as combine_first
    del df[i + '_y']  # left (block 1) is dominant
    df = df.rename(columns={i + '_x': i})
del df_b1
del df_b2

# b1 + b2 + income-gen
df = df.merge(b2_income_gen, on='ID', how='left', validate='one_to_one')
del b2_income_gen

# b1 + b2 + kids
df = df.merge(b2_kids, on='ID', how='left', validate='one_to_one')
del b2_kids

# b1 + b2: mismatched labels

# b1 + b2: redundant columns
for i in ['Kawasan', 'No_IR', 'No_AIR', 'Perhubungan_KIR', 'Penerima_Pendapatan']:
    del df[i]

# b3: group expenditure by items
cons_two_digits = df_b3.groupby(['ID', 'TwoD'])['Amaun'].sum().reset_index()
cons_two_digits = pd.pivot(cons_two_digits, index='ID', columns='TwoD', values='Amaun').reset_index()  # long to wide

# b3: rename item 13 ID
b3_item13 = df_b3_item13.rename(columns={'id': 'ID'})

# b3: group expenditure for item 13
col_item13 = [i for i in b3_item13.columns if ('ID' not in i)]
b3_item13[13] = b3_item13[col_item13].sum(axis=1)
b3_item13 = b3_item13[['ID', 13]]

# b3: combine items 1-12 and item 13 in wide format
cons_two_digits = cons_two_digits.merge(b3_item13, on='ID', how='left', validate='one_to_one')

# b3: rename columns in wide format
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

# b1 + b2 + b3
df = df.merge(cons_two_digits, on='ID', how='left', validate='one_to_one')
del df_b3
del cons_two_digits
del b3_item13
del df_b3_item13

# Total spending
df['cons_01_13'] = df['Jum_Perbelanjaan'] + df['cons_13']

# Margins
df['gross_margin'] = (df['INCS09'] / 12) - df['Jum_Perbelanjaan']
# df['net_margin'] = (df['INCS08_hh'] / 12) - df['Jum_Perbelanjaan']

# Rename columns to be more intuitive
dict_rename = \
    {
        'ID': 'id',
        'Negeri': 'state',
        'Strata': 'urban',
        'Kump_Etnik': 'ethnicity',
        'Jum_IR': 'hh_size',
        # 'Type_LQ': 'house_type',
        # 'LQ_owned': 'house_status',
        'Wajaran': 'svy_weight',
        'Jum_Perbelanjaan': 'cons_01_12',
        # 'cons_01_13': 'cons_01_13',
        'Monthly_Inc': 'monthly_income',
        'INCS01': 'salaried_wages',
        'INCS02': 'other_wages',
        'INCS03': 'asset_income',
        'INCS05': 'gross_transfers',
        # 'INCS06_hh': 'net_transfers',
        # 'INCS07_hh': 'gross_income',
        # 'INCS08_hh': 'net_income',
        'INCS09': 'gross_income',
        # 'gross_margin': 'gross_margin',
        # 'net_margin': 'net_margin',
        'Jantina': 'male',
        'Umur': 'age',
        'Taraf_Perkahwinan': 'marriage',
        'Pencapaian_Pendidikan': 'education_basic',
        'Sijil_Tertinggi': 'education',
        'Taraf_Aktiviti': 'emp_status',
        # 'Penerima_Pendapatan': 'receives_income',
        'Pekerjaan': 'occupation',
        'Industri': 'industry',
        'Kewarganegaraan': 'malaysian',
        # 'income_gen_members': 'income_gen_members',
        # 'child': '',
        # 'adolescent': '',
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
    }
df = df.rename(columns=dict_rename)

# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# post-merge: convert income, expenditure, and margins into monthly per capita
for i in ['salaried_wages', 'other_wages', 'asset_income',
          'gross_transfers', 'gross_income']:
    df[i] = df[i] * df['hh_size'] / 12
for i in ['cons_01_12', 'cons_01_13'] + \
    ['cons_0' + str(i) for i in range(1, 10)] + \
    ['cons_' + str(i) for i in range(11, 14)]:
    df[i] = df[i] / df['hh_size']

# Drop NAs
df = df.dropna(axis=0, how='any')

# Harmonise dtypes
dict_dtypes_14 = \
    {
        'hh_size': 'int',
        'state': 'str',
        'urban': 'int',
        'ethnicity': 'str',
        'salaried_wages': 'float',
        'other_wages': 'float',
        'asset_income': 'float',
        'gross_transfers': 'float',
        'gross_income': 'float',
        'cons_01_12': 'float',
        'male': 'int',
        'age': 'int',
        'malaysian': 'int',
        'marriage': 'str',
        'education_basic': 'str',
        'education': 'str',
        'emp_status': 'str',
        'industry': 'str',
        'occupation': 'str',
        'income_gen_members': 'int',
        'child': 'int',
        'adolescent': 'int',
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
        'cons_01_13': 'float',
        'gross_margin': 'float',
    }
df = df.astype(dict_dtypes_14)

# V --- Save output
df.to_parquet(path_data + 'hies_2014_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2014: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
