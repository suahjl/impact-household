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
path_data = './data/hies_2016/'

# I --- Load b1
df_b1 = pd.read_stata(path_data + 'HES 2016- Household Records-BNM.dta')
# df_b1.to_parquet(path_data + 'blok1.parquet', compression='brotli')

# II --- Load b2
df_b2 = pd.read_stata(path_data + 'HES 2016- Member Records-BNM.dta')
# df_b2.to_parquet(path_data + 'blok2.parquet', compression='brotli')

# III --- Load b3
df_b3 = pd.read_stata(path_data + 'HES 2016- Expenditure Records-BNM.dta')
# df_b3.to_parquet(path_data + 'blok3.parquet', compression='brotli')

# IV --- Clean & Consolidate

# b2: Recode dummies
df_b2.loc[df_b2['Inc_recipient'] == 2, 'Inc_recipient'] = 0  # receives income
df_b2.loc[df_b2['Citizenship'] == 2, 'Citizenship'] = 0  # malaysian
df_b2.loc[df_b2['Gender'] == 2, 'Gender'] = 0  # male

# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('id')['Inc_recipient']\
    .sum()\
    .reset_index()\
    .rename(columns={'Inc_recipient': 'income_gen_members'})

# b2: Separate column for <= 12 year-olds, and <= 17 year-olds
b2_kids = df_b2[['id', 'Age']].copy()
b2_kids['Age'] = b2_kids['Age'].astype('int')
b2_kids.loc[b2_kids['Age'] <= 12, 'child'] = 1
b2_kids.loc[b2_kids['Age'] <= 17, 'underage'] = 1
for i in ['child', 'underage']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('id')[['child', 'underage']].sum().reset_index()

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

# b1 + b2 + income-gen
df = df.merge(b2_income_gen, on='id', how='left', validate='one_to_one')
del b2_income_gen

# b1 + b2 + kids
df = df.merge(b2_kids, on='id', how='left', validate='one_to_one')
del b2_kids

# b1 + b2: mismatched labels
del df['No_Mem_HH']

# b1 + b2: redundant columns
for i in ['Region', 'HH_Mem_No', 'Relationship', 'Schooling']:
    del df[i]

# b3: generate new column indicating 2D category
df_b3['TwoD'] = df_b3['FourD'].str[:2]

# b3: group expenditure by items
cons_two_digits = df_b3.groupby(['id', 'TwoD'])['Amount'].sum().reset_index()
cons_two_digits = pd.pivot(cons_two_digits, index='id', columns='TwoD', values='Amount').reset_index()  # long to wide
cons_two_digits = cons_two_digits.rename(
    columns={
        '01': 'cons_01',
        '02': 'cons_02',
        '03': 'cons_03',
        '04': 'cons_04',
        '05': 'cons_05',
        '06': 'cons_06',
        '07': 'cons_07',
        '08': 'cons_08',
        '09': 'cons_09',
        '10': 'cons_10',
        '11': 'cons_11',
        '12': 'cons_12',
        '13': 'cons_13',
    }
)
cons_two_digits = cons_two_digits.fillna(0)

# b1 + b2 + b3
df = df.merge(cons_two_digits, on='id', how='left', validate='one_to_one')
del df_b3
del cons_two_digits

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
        'Gender': 'male',
        'Age': 'age',
        'Marital_Status': 'marriage',
        'Highest_Level_Edu': 'education_detailed',
        'Highest_Certificate': 'education',
        'Act_Status': 'emp_status',
        'Inc_recipient': 'receives_income',
        'Occupation': 'occupation',
        'Industry': 'industry',
        'Citizenship': 'malaysian',
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
df.to_parquet(path_data + 'hies_2016_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2016: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
