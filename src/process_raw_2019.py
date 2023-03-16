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
path_data = './data/hies_2019/'

# I --- Load b1
df_b1 = pd.read_excel(path_data + 'Blok 1.xls')
# df_b1.to_parquet(path_data + 'blok1.parquet', compression='brotli')

# II --- Load b2
df_b2 = pd.read_excel(path_data + 'Blok 2.xls', sheet_name='Blok 2')
for i in tqdm(['Blok 22', 'Blok 23', 'Blok 24']):
    d = pd.read_excel(path_data + 'Blok 2.xls', sheet_name=i, header=None)
    d.columns = list(df_b2.columns)
    df_b2 = pd.concat([df_b2, d], axis=0)
    del d
# df_b2.to_parquet(path_data + 'blok2.parquet', compression='brotli')

# III --- Load b3
df_b3_sheet1 = pd.read_excel(path_data + 'Blok 3.xls', sheet_name='Blok 3')  # load first sheet with col names
dict_file_b3 = pd.read_excel(path_data + 'Blok 3.xls', sheet_name=None, header=None)  # load entire file
del dict_file_b3['Blok 3']  # remove first sheet with col names
df_b3 = pd.concat(dict_file_b3.values(), ignore_index=True)
df_b3.columns = list(df_b3_sheet1.columns)
df_b3 = pd.concat([df_b3_sheet1, df_b3], axis=0)
df_b3 = df_b3.reset_index(drop=True)
del df_b3_sheet1
del dict_file_b3
# df_b3.to_parquet(path_data + 'blok3.parquet', compression='brotli')

# IV --- Clean & Consolidate

# b2: Recode dummies
df_b2.loc[df_b2['PP'] == 2, 'PP'] = 0  # receives income
df_b2.loc[df_b2['KW'] == 2, 'KW'] = 0  # malaysian
df_b2.loc[df_b2['J'] == 2, 'J'] = 0  # male
df_b1.loc[df_b1['Strata'] == 2, 'Strata'] = 0  # urban

# b1 & b2: Recode ethnicity
dict_ethnicity = \
    {
        1: 'bumiputera',
        2: 'chinese',
        3: 'indian',
        4: 'others'
    }
df_b1['Etnik'] = df_b1['Etnik'].replace(dict_ethnicity)
df_b2['Etnik'] = df_b2['Etnik'].replace(dict_ethnicity)

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
dict_house_status = \
    {
        1: 'owned',
        2: 'rented',
        3: 'squatters_owned',
        4: 'squatters_rented',
        5: 'quarters',
        6: 'others',
    }
df_b1['JMR'] = df_b1['JMR'].replace(dict_house_status)

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
df_b2['TP'] = df_b2['TP'].replace(dict_marriage_status)

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
df_b2['SJ'] = df_b2['SJ'].replace(dict_education)

# b2: Recode emp_status
emp_detailed = False
if not emp_detailed:
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
            10: 'pensioner',
            11: 'others',
            12: 'others',
            13: 'child_not_at_work',
            14: 'child_not_at_work',
            15: 'others',
        }
elif emp_detailed:  # 2019 HIES has richer categories
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
            9: 'gov_pensioner',
            10: 'priv_pensioner',
            11: 'elderly',
            12: 'oku',
            13: 'child_not_at_school',
            14: 'infant',
            15: 'others',
        }
df_b2['TA'] = df_b2['TA'].replace(dict_emp_status)

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
    }
df_b2['PK'] = df_b2['PK'].replace(dict_occ)

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
        22: 'others'
    }
df_b2['IND'] = df_b2['IND'].replace(dict_ind)

# b2: age group
df_b2.loc[(df_b2['U'] <= 29), 'age_group'] = '0_29'
df_b2.loc[((df_b2['U'] >= 30) & (df_b2['U'] <= 39)), 'age_group'] = '30_39'
df_b2.loc[((df_b2['U'] >= 40) & (df_b2['U'] <= 49)), 'age_group'] = '40_49'
df_b2.loc[((df_b2['U'] >= 50) & (df_b2['U'] <= 59)), 'age_group'] = '50_59'
df_b2.loc[((df_b2['U'] >= 60) & (df_b2['U'] <= 69)), 'age_group'] = '60_69'
df_b2.loc[(df_b2['U'] >= 70), 'age_group'] = '70+'

# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('HID')['PP'].sum().reset_index().rename(columns={'PP': 'income_gen_members'})

# b2: Separate column for <= 12 year-olds, and <= 17 year-olds
b2_kids = df_b2[['HID', 'U']].copy()
b2_kids.loc[b2_kids['U'] <= 12, 'child'] = 1
b2_kids.loc[(b2_kids['U'] > 12) & (b2_kids['U'] <= 17), 'adolescent'] = 1
for i in ['child', 'adolescent']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('HID')[['child', 'adolescent']].sum().reset_index()

# b2: Keep only head of households
print(tabulate(pd.crosstab(df_b2['PP'], df_b2['PKIR']), showindex=True, headers='keys', tablefmt="pretty"))
df_b2 = df_b2[df_b2['PKIR'] == 1].copy()

# b1: Compute gross income (ALL INCOME + GROSS TRANSFERS)
df_b1['INCS07_hh'] = df_b1['INCS01_hh'] + df_b1['INCS02_hh'] + df_b1['INCS03_hh'] + df_b1['INCS05_hh']

# b1 + b2
col_overlap = [i for i in df_b1.columns if i in df_b2.columns and ('HID' not in i)]
df = df_b1.merge(df_b2, on='HID', how='left', validate='one_to_one')
for i in tqdm(col_overlap):
    df.loc[df[i + '_x'].isna(), i + '_x'] = df[i + '_y']  # same as combine_first
    del df[i + '_y']  # left (block 1) is dominant
    df = df.rename(columns={i + '_x': i})
del df_b1
del df_b2

# b1 + b2 + income-gen
df = df.merge(b2_income_gen, on='HID', how='left', validate='one_to_one')
del b2_income_gen

# b1 + b2 + kids
df = df.merge(b2_kids, on='HID', how='left', validate='one_to_one')
del b2_kids

# b1 + b2: mismatched labels
del df['NOAIR']

# b1 + b2: redundant columns
for i in ['NOIR', 'PKIR', 'PP']:
    del df[i]

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

# b1 + b2 + b3
df = df.merge(cons_two_digits, on='HID', how='left', validate='one_to_one')
del df_b3
del cons_two_digits

# Margins
df['gross_margin'] = (df['INCS07_hh'] / 12) - df['Jumlah_perbelanjaan_01_12_sebula']
df['net_margin'] = (df['INCS08_hh'] / 12) - df['Jumlah_perbelanjaan_01_12_sebula']

# Rename columns to be more intuitive
dict_rename = \
    {
        'HID': 'id',
        'NOIR': 'member_no',
        'Wajaran': 'svy_weight',
        # 'hh_size': 'hh_size',
        # 'income_gen_members': 'income_gen_members',
        'Etnik': 'ethnicity',
        'Negeri': 'state',
        'Strata': 'urban',
        'INCS01_hh': 'salaried_wages',
        'INCS02_hh': 'other_wages',
        'INCS03_hh': 'asset_income',
        'INCS05_hh': 'gross_transfers',
        'INCS06_hh': 'net_transfers',
        'INCS07_hh': 'gross_income',
        'INCS08_hh': 'net_income',
        # 'gross_margin': 'gross_margin',
        # 'net_margin': 'net_margin',
        'JMR': 'house_status',
        'Jumlah_perbelanjaan_01_12_sebula': 'cons_01_12',
        'Jumlah_perbelanjaan_01_13_sebula': 'cons_01_13',
        'Jumlah_pendapatan_sebulan': 'monthly_income',
        'PKIR': 'member_relation',
        'J': 'male',
        'U': 'age',
        # 'age_group': 'age_group',
        'KW': 'malaysian',
        'TP': 'marriage',
        # 'PP': 'receives_income',
        'TA': 'emp_status',
        'IND': 'industry',
        'PK': 'occupation',
        'SJ': 'education',
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
          'gross_transfers', 'net_transfers', 'gross_income', 'net_income']:
    df[i] = df[i] * df['hh_size'] / 12
for i in ['cons_01_12', 'cons_01_13'] + \
    ['cons_0' + str(i) for i in range(1, 10)] + \
    ['cons_' + str(i) for i in range(11, 14)]:
    df[i] = df[i] / df['hh_size']

# Drop NAs
df = df.dropna(axis=0, how='any')

# Harmonise dtypes
dict_dtypes_19 = \
    {
        'state': 'str',
        'education': 'str',
        'ethnicity': 'str',
        'malaysian': 'int',
        'income_gen_members': 'int',
        'adolescent': 'int',
        'child': 'int',
        'male': 'int',
        'age_group': 'str',
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
        'net_income': 'float',
        'cons_01_12': 'float',
        'cons_01_13': 'float',
        'monthly_income': 'float',
        'gross_income': 'float',
        'age': 'int',
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
        'gross_margin': 'float',
        'net_margin': 'float',
    }
df = df.astype(dict_dtypes_19)

# V --- Save output
df.to_parquet(path_data + 'hies_2019_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2019: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
