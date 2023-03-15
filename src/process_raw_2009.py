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

# II --- b2 (not necessary)

# III --- b3 (not necessary)

# IV --- Clean & Consolidate

# Set column names
df.columns = df.loc[2]  # 3rd row
df = df.loc[3:].reset_index(drop=True)

# Recode dummies

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
        'INCS06': 'net_transfers',
        'INCS07': 'gross_income',
        'INCS08': 'net_income',
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
        # 'underage': '',
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

# Drop crazy lines (because of excel + manual intervention by past analysts)
df = df[~(df['id'].isna())].copy()

# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# V --- Save output
df.to_parquet(path_data + 'hies_2009_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2009: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
