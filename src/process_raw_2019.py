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

# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('HID')['PP'].sum().reset_index().rename(columns={'PP': 'income_gen_members'})

# b2: Separate column for <= 12 year-olds, and <= 17 year-olds
b2_kids = df_b2[['HID', 'U']].copy()
b2_kids.loc[b2_kids['U'] <= 12, 'child'] = 1
b2_kids.loc[b2_kids['U'] <= 17, 'underage'] = 1
for i in ['child', 'underage']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('HID')[['child', 'underage']].sum().reset_index()

# b2: Keep only head of households
print(tabulate(pd.crosstab(df_b2['PP'], df_b2['PKIR']), showindex=True, headers='keys', tablefmt="pretty"))
df_b2 = df_b2[df_b2['PKIR'] == 1].copy()

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
del df['NOIR']
del df['PKIR']

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
        'Strata': 'strata',
        'INCS01_hh': 'salaried_wages',
        'INCS02_hh': 'other_wages',
        'INCS03_hh': 'asset_income',
        'INCS05_hh': 'gross_transfers',
        'INCS06_hh': 'net_transfers',
        'INCS08_hh': 'net_income',
        'JMR': 'house_type',
        'Jumlah_perbelanjaan_01_12_sebula': 'cons_01_12',
        'Jumlah_perbelanjaan_01_13_sebula': 'cons_01_13',
        'Jumlah_pendapatan_sebulan': 'monthly_income',
        'PKIR': 'member_relation',
        'J': 'male',
        'U': 'age',
        'KW': 'malaysian',
        'TP': 'marriage',
        'PP': 'receives_income',
        'TA': 'emp_status',
        'IND': 'industry',
        'PK': 'occupation',
        'SJ': 'education',
        # 'child': '',
        # 'underage': '',
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

# V --- Save output
df.to_parquet(path_data + 'hies_2019_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2019: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
