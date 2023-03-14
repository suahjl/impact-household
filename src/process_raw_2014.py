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
path_data = './data/hies_2014/'

# I --- Load b1
df_b1 = pd.read_csv(path_data + 'Blok1.txt', sep=',')
df_b1.to_parquet(path_data + 'blok1.parquet', compression='brotli')

# II --- Load b2
df_b2 = pd.read_csv(path_data + 'Blok2.txt', sep=',')
df_b2.to_parquet(path_data + 'blok2.parquet', compression='brotli')

# III --- Load b3 (NOT NEEDED AS YET)
if skip_b3:
    pass
# elif not skip_b3:
#     df_b3 = pd.read_stata(path_data + 'Blok3.txt', sep=',')\
#     df_b3.to_parquet(path_data + 'blok3.parquet', compression='brotli')

# IV --- Clean & Consolidate
# b2: Keep only head of households
print(tabulate(pd.crosstab(df_b2['Inc_recipient'], df_b2['Relationship']), showindex=True, headers='keys',
               tablefmt="pretty"))
df_b2 = df_b2[df_b2['Relationship'] == '01'].copy()
# b1 + b2
col_overlap = [i for i in df_b1.columns if i in df_b2.columns and ('id' not in i)]
df = df_b1.merge(df_b2, on='id', how='left', validate='one_to_one')
for i in tqdm(col_overlap):
    df.loc[df[i + '_x'].isna(), i + '_x'] = df[i + '_y']  # same as combine_first
    del df[i + '_y']  # left (block 1) is dominant
    df = df.rename(columns={i + '_x': i})
del df_b1
del df_b2
# b1 + b2: mismatched labels
del df['No_Mem_HH']
# b1 + b2: redundant columns
for i in ['Region', 'HH_Mem_No', 'Relationship', 'Schooling']:
    del df[i]
# b1 + b2 + b3 (WIP)
if skip_b3:
    print('Not merging b3')
# elif not skip_b3:
#     print('Merging b3')
# Rename columns to be more intuitive
dict_rename = \
    {
        # 'id': 'id',
        'State': 'state',
        'Strata': 'strata',
        'Ethnic': 'ethnicity',
        'Tot_mem': 'hh_size',
        'Type_LQ': 'house_type',
        'LQ_owned': 'house_status',
        'Weight': 'svy_weight',
        'Total_Exp_01_12': 'cons_01_12',
        'Total_Exp_01_13': 'cons_01_13',
        'Monthly_Inc': 'monthly_income',
        'INCS01_hh': 'salaried_wages',
        'INCS02_hh': 'other_wages',
        'INCS03_hh': 'asset_income',
        'INCS05_hh': 'gross_transfers',
        'INCS06_hh': 'net_transfers',
        'INCS07_hh': 'gross_income',
        'INCS08_hh': 'net_income',
        'Gender': 'sex',
        'Age': 'age',
        'Marital_Status': 'marriage',
        'Highest_Level_Edu': 'education_detailed',
        'Highest_Certificate': 'education',
        'Act_Status': 'emp_status',
        'Inc_recipient': 'receives_income',
        'Occupation': 'occupation',
        'Industry': 'industry',
        'Citizenship': 'citizenship',
    }
df = df.rename(columns=dict_rename)
# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# V --- Save output
df.to_parquet(path_data + 'hies_2016_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2016: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
