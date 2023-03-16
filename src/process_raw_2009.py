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
path_data = './data/hies_2009/'

# I --- Load full data set (only 2009 has this pre-arranged)
df = pd.read_excel(path_data + 'HES200910_raw.xlsx', sheet_name='raw data_HES0910')

# II --- b2 (not available)

# III --- b3 (not available)

# IV --- Clean & Consolidate

# Set column names
df.columns = df.loc[2]  # 3rd row
df = df.loc[3:].reset_index(drop=True)

# Drop crazy lines (because of excel + manual intervention by past analysts)
df = df[~(df['ID'].isna())].copy()

# Recode dummies

# Recode state
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
df['state'] = df['state'].replace(dict_state)

# Recode marriage
dict_marriage_status = \
    {
        1: 'never',
        2: 'married',
        3: 'widowed',
        4: 'divorced',
        5: 'separated',
        9: 'never',  # if no info, assumed to be never married
    }
df['maritul_status'] = df['maritul_status'].replace(dict_marriage_status)

# Recode education (guesswork based on other vintages, since there's no documentation)
dict_education = \
    {
        1: 'preschool',
        2: 'primary',
        3: 'lower_sec',
        4: 'upper_sec',
        5: 'pre_uni',
        6: 'post_sec_non_tertiary',
        7: 'tertiary_diploma',
        8: 'tertiary_grad',
        9: 'tertiary_postdoc',
        99: 'no_educ',
    }
df['highest_certificate_obtained'] = df['highest_certificate_obtained'].replace(dict_education)

# Recode occupation
df['occupation'] = df['occupation'].astype('str').str[0]
df['occupation'] = df['occupation'].astype('int')
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
        0: 'others',
    }
df['occupation'] = df['occupation'].replace(dict_occ)

# b2: age group
df.loc[(df['age'] <= 29), 'age_group'] = '0_29'
df.loc[((df['age'] >= 30) & (df['age'] <= 39)), 'age_group'] = '30_39'
df.loc[((df['age'] >= 40) & (df['age'] <= 49)), 'age_group'] = '40_49'
df.loc[((df['age'] >= 50) & (df['age'] <= 59)), 'age_group'] = '50_59'
df.loc[((df['age'] >= 60) & (df['age'] <= 69)), 'age_group'] = '60_69'
df.loc[(df['age'] >= 70), 'age_group'] = '70+'

# Sketchy data types
for i in ['Tot_exp_01_13', 'wt', 'age',
          'F10', 'F21', 'F22', 'F23', 'F30',
          'F40', 'F50', 'F70', 'F80']:
    df[i] = df[i].astype('float')

# Combine income columns
# Salary and wages
df = df.rename(columns={'F10': 'INCS01'})
# Other wage income
df['INCS02'] = df[['F21', 'F70']].sum(axis=1)
for i in ['F21', 'F70']:
    del df[i]
# Asset / property income
df['INCS03'] = df[['F22', 'F23', 'F30']].sum(axis=1)  # rent occ + rent other + prop income
for i in ['F22', 'F23', 'F30']:
    del df[i]
# Gross transfers
df['INCS05'] = df[['F40', 'F50']].sum(axis=1)
for i in ['F40', 'F50']:
    del df[i]
# Gross income
df = df.rename(columns={'F80': 'INCS09'})

# Trim 4D / 6D items
for i in ['Group_1311',
       'Group_1312', 'Group_1313', 'item_131314', 'item_131101', 'item_131109',
       'item_131110', 'item_131114', 'item_131115', 'item_131122',
       'item_131123', 'item_131196', 'item_131198', 'item_131301',
       'item_131302', 'item_131303', 'item_131304', 'item_131305',
       'item_131306', 'item_131307', 'item_131308', 'item_131309',
       'item_131310', 'item_131311', 'item_131312', 'item_131313',
       'Group_1314', 'item_131315', 'item_131316', 'item_131317',
       'item_131318', 'item_131319', 'item_131398', 'item_131401',
       'item_131402', 'item_131403']:
    del df[i]

# Expenditure ex item 13
df['cons_01_12'] = df['Tot_exp_01_13'] - df['Group_13']

# Rename columns to be more intuitive
dict_rename = \
    {
        'ID': 'id',
        # 'state': 'state',
        # 'Strata': 'strata',
        # 'Kump_Etnik': 'ethnicity',
        # 'Jum_IR': 'hh_size',
        # 'Type_LQ': 'house_type',
        # 'LQ_owned': 'house_status',
        'wt': 'svy_weight',
        # 'Jum_Perbelanjaan': 'cons_01_12',
        'Tot_exp_01_13': 'cons_01_13',
        'Monthly_Inc': 'monthly_income',
        'INCS01': 'salaried_wages',
        'INCS02': 'other_wages',
        'INCS03': 'asset_income',
        'INCS05': 'gross_transfers',
        # 'INCS06': 'net_transfers',
        # 'INCS07': 'gross_income',
        # 'INCS08': 'net_income',
        'INCS09': 'gross_income',
        # 'Jantina': 'male',
        # 'age': 'age',
        'maritul_status': 'marriage',
        # 'Pencapaian_Pendidikan': 'education_detailed',
        'highest_certificate_obtained': 'education',
        # 'Taraf_Aktiviti': 'emp_status',
        # 'Penerima_Pendapatan': 'receives_income',
        # 'occupation': 'occupation',
        # 'Industri': 'industry',
        # 'Kewarganegaraan': 'malaysian',
        # 'child': '',
        # 'adolescent': '',
        'Group_01': 'cons_01',
        'Group_02': 'cons_02',
        'Group_03': 'cons_03',
        'Group_04': 'cons_04',
        'Group_05': 'cons_05',
        'Group_06': 'cons_06',
        'Group_07': 'cons_07',
        'Group_08': 'cons_08',
        'Group_09': 'cons_09',
        'Group_10': 'cons_10',
        'Group_11': 'cons_11',
        'Group_12': 'cons_12',
        'Group_13': 'cons_13',
    }
df = df.rename(columns=dict_rename)

# Fill missing values in expenditure items 1 to 13 with 0
for i in ['cons_0' + str(i) for i in range(1, 10)]:
    df[i] = df[i].fillna(0)
for i in ['cons_1' + str(i) for i in range(0, 4)]:
    df[i] = df[i].fillna(0)
for i in ['cons_01_12', 'cons_01_13']:
    df[i] = df[i].fillna(0)

# Margins
df['gross_margin'] = (df['gross_income'] / 12) - df['cons_01_12']
# df['net_margin'] = (df['net_income'] / 12) - df['cons_01_12']

# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# post-merge: convert income, expenditure, and margins into monthly per capita (not possible --- no hh_size)

# Column name
df.columns.name = None

# Drop NAs
df = df.dropna(axis=0, how='any')

# Harmonise dtypes
dict_dtypes_09 = \
    {
        'state': 'str',
        'salaried_wages': 'float',
        'other_wages': 'float',
        'asset_income': 'float',
        'gross_transfers': 'float',
        'cons_01_12': 'float',
        'cons_01_13': 'float',
        'gross_income': 'float',
        'age': 'int',
        'marriage': 'str',
        'occupation': 'str',
        'education': 'str',
        'age_group': 'str',
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
    }
df = df.astype(dict_dtypes_09)

# V --- Save output
df.to_parquet(path_data + 'hies_2009_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2009: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
