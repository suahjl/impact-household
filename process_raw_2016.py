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

# b1 & b2: Redefine some dtypes
for i in ['No_Mem_HH', 'Type_LQ']:
    df_b1[i] = df_b1[i].astype('int')
for i in ['HH_Mem_No', 'Relationship', 'Gender', 'Age',
          'Marital_Status', 'Schooling',
          'Highest_Level_Edu', 'Highest_Certificate',
          'Act_Status', 'Inc_recipient']:
    df_b2[i] = df_b2[i].astype('int')

# b1 & b2: Recode dummies
df_b2.loc[df_b2['Inc_recipient'] == 2, 'Inc_recipient'] = 0  # receives income
df_b2.loc[df_b2['Citizenship'] == 2, 'Citizenship'] = 0  # malaysian
df_b2.loc[df_b2['Gender'] == 2, 'Gender'] = 0  # male

df_b1['State'] = df_b1['State'].cat.codes

df_b1['Ethnic'] = df_b1['Ethnic'].cat.codes

df_b1['Strata'] = df_b1['Strata'].cat.codes + 1
df_b1.loc[df_b1['Strata'] == 2, 'Strata'] = 0  # urban / rural

df_b2['Occupation'] = df_b2['Occupation'].cat.codes
df_b2['Industry'] = df_b2['Industry'].cat.codes

# b1 & b2: Recode ethnicity
dict_ethnicity = \
    {
        0: 'bumiputera',
        1: 'chinese',
        2: 'indian',
        3: 'others'
    }
df_b1['Ethnic'] = df_b1['Ethnic'].replace(dict_ethnicity)
df_b2['Ethnic'] = df_b2['Ethnic'].replace(dict_ethnicity)

# b1: Recode state
dict_state = \
    {
        0: 'johor',
        1: 'kedah',
        2: 'kelantan',
        3: 'melaka',
        4: 'negeri_sembilan',
        5: 'pahang',
        6: 'penang',
        7: 'perak',
        8: 'perlis',
        9: 'selangor',
        10: 'terengganu',
        11: 'sabah',
        12: 'sarawak',
        13: 'kuala_lumpur',
        14: 'labuan',
        15: 'putrajaya',
    }
df_b1['State'] = df_b1['State'].replace(dict_state)

# b1: Recode housing type
dict_house_type = \
    {
        1: 'bungalow',
        2: 'semi_detached',
        3: 'terrace',
        4: 'longhouse',
        5: 'flat',
        6: 'apartment',
        7: 'condo',
        8: 'shophouse',
        9: 'room',
        10: 'hut',
        11: 'others',
    }
df_b1['Type_LQ'] = df_b1['Type_LQ'].replace(dict_house_type)

# b1: Recode housing status (only in 2016)
dict_house_status = \
    {
        1: 'owned',
        2: 'rented',
        3: 'squatters_owned',
        4: 'squatters_rented',
        5: 'quarters',
        6: 'others',
    }
df_b1['LQ_owned'] = df_b1['LQ_owned'].replace(dict_house_status)

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
df_b2['Marital_Status'] = df_b2['Marital_Status'].replace(dict_marriage_status)

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
df_b2['Highest_Certificate'] = df_b2['Highest_Certificate'].replace(dict_education)

# b2: Recode detailed education
dict_education = \
    {
        0: 'preschool',
        1: 'primary',
        2: 'lower_sec',
        3: 'upper_sec',
        4: 'pre_uni',
        5: 'post_sec_non_tertiary',
        6: 'tertiary_diploma',
        7: 'tertiary_grad',
        8: 'tertiary_postdoc',
        9: 'informal',
        10: 'no_educ',
        11: 'not_yet',
    }
df_b2['Highest_Level_Edu'] = df_b2['Highest_Level_Edu'].replace(dict_education)

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
df_b2['Act_Status'] = df_b2['Act_Status'].replace(dict_emp_status)

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
        0: 'others',
    }
df_b2['Occupation'] = df_b2['Occupation'].replace(dict_occ)

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
        0: 'others'
    }
df_b2['Industry'] = df_b2['Industry'].replace(dict_ind)

# b2: Separate column for number of income-generating members
b2_income_gen = df_b2.groupby('id')['Inc_recipient'] \
    .sum() \
    .reset_index() \
    .rename(columns={'Inc_recipient': 'income_gen_members'})

# b2: non-income-generating adult females (18-59)
b2_idle_women = df_b2[(((df_b2['Age'] >= 18) & (df_b2['Age'] < 60)) &
                       (df_b2['Inc_recipient'] == 0) & (df_b2[
                                                            'Gender'] == 0))].copy()  # keep only rows corresponding to adult females not working
b2_idle_women = b2_idle_women.groupby('id')['Inc_recipient'] \
    .count() \
    .reset_index() \
    .rename(columns={'Inc_recipient': 'non_working_adult_females'})

# b2: Income-generating adult females (18-59)
b2_working_women = df_b2[(((df_b2['Age'] >= 18) & (df_b2['Age'] < 60)) &
                          (df_b2['Inc_recipient'] == 1) & (df_b2[
                                                               'Gender'] == 0))].copy()  # keep only rows corresponding to working adult females
b2_working_women = b2_working_women.groupby('id')['Inc_recipient'] \
    .count() \
    .reset_index() \
    .rename(columns={'Inc_recipient': 'working_adult_females'})

# b2: Separate column for <= 12 year-olds, <= 17 year-olds, and elderly (>= 60 year olds)
b2_kids = df_b2[['id', 'Age']].copy()
b2_kids['Age'] = b2_kids['Age'].astype('int')
b2_kids.loc[b2_kids['Age'] <= 17, 'kid'] = 1
b2_kids.loc[b2_kids['Age'] <= 12, 'child'] = 1
b2_kids.loc[(b2_kids['Age'] > 12) & (b2_kids['Age'] <= 17), 'adolescent'] = 1
b2_kids.loc[b2_kids['Age'] >= 60, 'elderly'] = 1
b2_kids.loc[(b2_kids['Age'] >= 18) & (b2_kids['Age'] <= 64), 'working_age2'] = 1
b2_kids.loc[b2_kids['Age'] >= 65, 'elderly2'] = 1
b2_kids.loc[(b2_kids['Age'] >= 18) & (b2_kids['Age'] <= 59), 'working_age'] = 1
for i in ['kid', 'child', 'adolescent', 'elderly', 'working_age2', 'elderly2', 'working_age']:
    b2_kids.loc[b2_kids[i].isna(), i] = 0
b2_kids = b2_kids.groupby('id')[
    ['kid', 'child', 'adolescent', 'elderly', 'working_age2', 'elderly2', 'working_age']].sum().reset_index()

# b2: Keep only head of households
print(tabulate(pd.crosstab(df_b2['Inc_recipient'], df_b2['Relationship']), showindex=True, headers='keys',
               tablefmt="pretty"))
df_b2 = df_b2[df_b2['Relationship'] == 1].copy()

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
df.loc[df['income_gen_members'].isna(), 'income_gen_members'] = 0

# b1 + b2 + kids
df = df.merge(b2_kids, on='id', how='left', validate='one_to_one')
del b2_kids
for i in ['child', 'adolescent', 'elderly']:
    df.loc[df[i].isna(), i] = 0

# b1 + b2 + idle adult women
df = df.merge(b2_idle_women, on='id', how='left', validate='one_to_one')
del b2_idle_women
df.loc[df['non_working_adult_females'].isna(), 'non_working_adult_females'] = 0

# b1 + b2 + working adult women
df = df.merge(b2_working_women, on='id', how='left', validate='one_to_one')
del b2_working_women
df.loc[df['working_adult_females'].isna(), 'working_adult_females'] = 0

# b1 + b2: mismatched labels
del df['No_Mem_HH']

# b1 + b2: redundant columns
for i in ['Region', 'HH_Mem_No', 'Relationship', 'Schooling', 'Inc_recipient']:
    del df[i]

# b3: generate new column indicating 2D category
df_b3['TwoD'] = df_b3['FourD'].str[:2]

# b3: isolate special subitems
# Fuel only: item 0722
cons_fuel = df_b3[df_b3['FourD'].isin(['0722', '0451'])]
cons_fuel = cons_fuel.groupby(['id', 'FourD'])['Amount'].sum().reset_index()
cons_fuel = pd.pivot(cons_fuel, index='id', columns='FourD', values='Amount').reset_index()  # long to wide
cons_fuel = cons_fuel.rename(
    columns={
        '0722': 'cons_0722_fuel',
        '0451': 'cons_0451_elec'
    }
)
# Transport ex cars, motorcycles, bicycles, and servicing
cons_transport_ex_bigticket = \
    df_b3[~(df_b3['FourD'].isin(
        ['0711', '0712', '0713', '0723']
    )) & (df_b3['TwoD'] == '07')]  # exclude cars, motorcycles, bicycles, and servicing
cons_transport_ex_bigticket = cons_transport_ex_bigticket.groupby(['id', 'TwoD'])['Amount'].sum().reset_index()
cons_transport_ex_bigticket = \
    pd.pivot(cons_transport_ex_bigticket, index='id', columns='TwoD', values='Amount').reset_index()  # long to wide
cons_transport_ex_bigticket = cons_transport_ex_bigticket.rename(
    columns={
        '07': 'cons_07_ex_bigticket'
    }
)

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
df = df.merge(cons_fuel, on='id', how='left', validate='one_to_one')
df = df.merge(cons_transport_ex_bigticket, on='id', how='left', validate='one_to_one')
del df_b3
del cons_two_digits
del cons_fuel
del cons_transport_ex_bigticket

# Margins
df['gross_margin'] = (df['INCS07_hh'] / 12) - df['Total_Exp_01_12']
df['net_margin'] = (df['INCS08_hh'] / 12) - df['Total_Exp_01_12']

# Rename columns to be more intuitive
dict_rename = \
    {
        # 'id': 'id',
        'State': 'state',
        'Strata': 'urban',
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
        # 'gross_margin': 'gross_margin',
        # 'net_margin': 'net_margin',
        'Gender': 'male',
        'Age': 'age',
        'Marital_Status': 'marriage',
        'Highest_Level_Edu': 'education_detailed',
        'Highest_Certificate': 'education',
        'Act_Status': 'emp_status',
        # 'Inc_recipient': 'receives_income',
        # 'income_gen_members': 'income_gen_members',
        # 'working_adult_females': '',
        # 'non_working_adult_females': '',
        'Occupation': 'occupation',
        'Industry': 'industry',
        'Citizenship': 'malaysian',
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
        # 'cons_0451_elec': ''
    }
df = df.rename(columns=dict_rename)

# Reset index + simplify IDs
if not (len(df.id.unique()) == len(df)): raise NotImplementedError('IDs are not unique')
df['id'] = df.reset_index().index
df = df.reset_index(drop=True)

# post-merge: convert income, expenditure, and margins into monthly per capita
for i in ['salaried_wages', 'other_wages', 'asset_income',
          'gross_transfers', 'net_transfers', 'gross_income', 'net_income']:
    # df[i] = df[i] / (12)
    df[i] = df[i] / (df['hh_size'] * 12)
for i in ['cons_01_12', 'cons_01_13'] + \
         ['cons_0' + str(i) for i in range(1, 10)] + \
         ['cons_' + str(i) for i in range(11, 14)] + \
         ['cons_0722_fuel', 'cons_0451_elec', 'cons_07_ex_bigticket']:
    df[i] = df[i] / df['hh_size']
for i in ['net_margin', 'gross_margin']:
    df[i] = df[i] / df['hh_size']

# post-merge: birth year
df['birth_year'] = 2016 - df['age']

# Drop NAs
df = df.dropna(axis=0, how='any')

# Harmonise dtypes
dict_dtypes_16 = \
    {
        'state': 'str',
        'urban': 'int',
        'ethnicity': 'str',
        'hh_size': 'int',
        'house_type': 'str',
        'house_status': 'str',
        'monthly_income': 'float',
        'cons_01_12': 'float',
        'cons_01_13': 'float',
        'salaried_wages': 'float',
        'other_wages': 'float',
        'asset_income': 'float',
        'gross_transfers': 'float',
        'net_transfers': 'float',
        'gross_income': 'float',
        'net_income': 'float',
        'male': 'int',
        'age': 'int',
        'birth_year': 'int',
        'marriage': 'str',
        'education_detailed': 'str',
        'education': 'str',
        'emp_status': 'str',
        'occupation': 'str',
        'industry': 'str',
        'malaysian': 'int',
        'income_gen_members': 'int',
        'working_adult_females': 'int',
        'non_working_adult_females': 'int',
        'kid': 'int',
        'child': 'int',
        'adolescent': 'int',
        'elderly': 'int',
        'working_age2': 'int',
        'working_age': 'int',
        'elderly2': 'int',
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
        'net_margin': 'float',
    }
df = df.astype(dict_dtypes_16)

# V --- Save output
df.to_parquet(path_data + 'hies_2016_consol.parquet', compression='brotli')

# VI --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_raw_2016: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
