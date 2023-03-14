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
skip_b3 = ast.literal_eval(os.getenv('IGNORE_BLOCK3'))
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/hies_2019/'

# I --- Load b1
df_b1 = pd.read_excel(path_data + 'Blok 1.xls')
df_b1.to_parquet(path_data + 'blok1.parquet', compression='brotli')

# II --- Load b2
df_b2 = pd.read_excel(path_data + 'Blok 2.xls', sheet_name='Blok 2')
for i in tqdm(['Blok 22', 'Blok 23', 'Blok 24']):
    d = pd.read_excel(path_data + 'Blok 2.xls', sheet_name=i, header=None)
    d.columns = list(df_b2.columns)
    df_b2 = pd.concat([df_b2, d], axis=0)
    del d
df_b2.to_parquet(path_data + 'blok2.parquet', compression='brotli')

# III --- Load b3 (NOT NEEDED AS YET)
if skip_b3:
    pass
# elif not skip_b3:
#     df_b3 = pd.read_excel(path_data + 'Blok 3.xls', sheet_name='Blok 3')
#     for i in tqdm(['Blok 3' + str(i) for i in range(2, 220)]):
#         d = pd.read_excel(path_data + 'Blok 3.xls', sheet_name=i, header=None)
#         d.columns = list(df_b3.columns)
#         df_b3 = pd.concat([df_b3, d], axis=0)
#         del d
#     df_b3.to_parquet(path_data + 'blok3.parquet', compression='brotli')

# IV --- Clean & Consolidate
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
# b1 + b2: mismatched labels
del df['NOAIR']
# b1 + b2: redundant columns
del df['NOIR']
del df['PKIR']
# b1 + b2 + b3 (WIP)
if skip_b3:
    print('Not merging b3')
# elif not skip_b3:
#     print('Merging b3')
# Rename columns to be more intuitive
dict_rename = \
    {
        'HID': 'id',
        'NOIR': 'member_no',
        'Wajaran': 'svy_weight',
        # 'hh_size': 'hh_size',
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
        'J': 'sex',
        'U': 'age',
        'KW': 'citizenship',
        'TP': 'marriage',
        'PP': 'receives_income',
        'TA': 'emp_status',
        'IND': 'industry',
        'PK': 'occupation',
        'SJ': 'education',
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
