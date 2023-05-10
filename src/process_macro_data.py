import pandas as pd
import numpy as np
from src.helper import \
    telsendmsg, telsendimg, telsendfiles, \
    get_data_from_api_ceic, get_data_from_ceic
from datetime import timedelta, date
from tabulate import tabulate
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast
import re

time_start = time.time()

# 0 --- Main settings
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')
path_data = './data/macro/'
# path_cpb = 'https://www.cpb.nl/sites/default/files/omnidownload/CPB-World-Trade-Monitor-February-2023.xlsx'

# I.A --- Load CEIC data
# List of ceic series ids and names
seriesids = pd.read_csv(path_data + 'ceic_macro.csv')
seriesids_petrol = pd.read_csv(path_data + 'ceic_macro_petrol.csv')
# Split into GDP (with historical extension) and others (without historical extension)
seriesids_gdp = seriesids[0:6]  # first 6
seriesids_others = seriesids[6:]
# Extract only series ID
seriesids_gdp = [re.sub("[^0-9]+", "", i) for i in list(seriesids_gdp['series_id'])]  # keep only numbers
seriesids_gdp = [int(i) for i in seriesids_gdp]
# Extract only series ID (others)
seriesids_others = [re.sub("[^0-9]+", "", i) for i in list(seriesids_others['series_id'])]  # keep only numbers
seriesids_others = [int(i) for i in seriesids_others]
# Extract only series ID (petrol)
# seriesids_petrol = [i for i in seriesids_petrol['series_id']]
seriesids_petrol = [int(i) for i in seriesids_petrol['series_id']]
# Pull data from CEIC
df_gdp = get_data_from_ceic(
    series_ids=seriesids_gdp,
    start_date=date(2005, 1, 1),
    historical_extension=True  # with historical extension
)
df_gdp = df_gdp.pivot(index='date', columns='name', values='value')
df_others = get_data_from_ceic(
    series_ids=seriesids_others,
    start_date=date(2005, 1, 1),
    historical_extension=False  # without historical extensions
)
df_others = df_others.pivot(index='date', columns='name', values='value')
df_petrol = get_data_from_ceic(
    series_ids=seriesids_petrol,
    start_date=date(2014, 12, 1),
    historical_extension=False
)
df_petrol = df_petrol.pivot(index='date', columns='name', values='value')
# Convert petrol prices into quarterly, and merge old and new series
df_petrol = df_petrol.reset_index()
df_petrol['quarter'] = pd.to_datetime(df_petrol['date']).dt.to_period('q')
df_petrol = \
    df_petrol.groupby('quarter')[['oil_price_ceiling_price_ron_95', '_dc_oil_price_ron95']] \
        .mean(numeric_only=True) \
        .reset_index()
df_petrol['oil_price_ceiling_price_ron_95'] = \
    df_petrol['oil_price_ceiling_price_ron_95']\
        .combine_first(df_petrol['_dc_oil_price_ron95'])
df_petrol['date'] = pd.to_datetime(df_petrol['quarter'].astype('str')).dt.date
df_petrol = df_petrol[['date', 'oil_price_ceiling_price_ron_95']]
df_petrol = df_petrol.set_index('date')
# Merge into single data frame
df = df_gdp.merge(df_others, how='outer', right_index=True, left_index=True)
df = df.merge(df_petrol, how='outer', right_index=True, left_index=True)
# Extract column names for later
cols_raw_snake = list(df.columns)
# Sort and reset index
df = df.sort_index().reset_index()
# Convert dates into datetime.date
df.columns.name = None
df['date'] = pd.to_datetime(df['date']).dt.date
# Delete interim data frames
del df_gdp
del df_others
del df_petrol
# Convert into quarterly series
df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q')
df = df.groupby('quarter')[cols_raw_snake].mean(numeric_only=True)  # this will also remove the 'date' column
# Custom series names
dict_nice_col_names = {
    'gdp_2015p_sa_exports_of_goods_services': 'ex',
    'gdp_2015p_sa_final_consumption_expenditure_government': 'gc',
    'gdp_2015p_sa_final_consumption_expenditure_private': 'pc',
    'gdp_2015p_sa_gross_fixed_capital_formation': 'gfcf',
    'gdp_2015p_sa_imports_of_goods_services': 'im',
    'gross_domesticproduct_gdp_2015p_sa': 'gdp',
    'commodity_price_nominal_energy_crude_oil_brent': 'brent',
    'consumer_price_index_cpi_': 'cpi',
    'cpi_transport_tp_': 'cpitransport',
    'consumer_price_index_cpi_core': 'cpicore',
    'economic_policy_uncertainty_index_global_ppp_adjusted_gdp': 'gepu',
    'forex_month_average_malaysian_ringgit_to_us_dollar': 'myrusd',
    'government_securities_yield_10_years': 'mgs10y',
    'nominal_effective_exchange_rate_index_bis_2020_100_broad': 'neer',
    'real_effective_exchange_rate_index_bis_2020_100_broad': 'reer',
    'short_term_interest_rate_month_end_klibor_3_months': 'klibor3m',
    'interbank_offered_rate_fixing_1_month': 'klibor1m',
    'oil_price_ceiling_price_ron_95': 'ron95'
}
df = df.rename(columns=dict_nice_col_names)

# I.B --- Load CPB data
df_cpb = pd.read_excel(path_data + 'CPB-World-Trade-Monitor-February-2023.xlsx', sheet_name='inpro_out')
df_cpb = df_cpb.iloc[[6, 24]]  # world row
df_cpb = df_cpb.loc[:, 'Unnamed: 5':]
list_months = list(pd.date_range(start=date(2000, 1, 1), periods=len(df_cpb.columns), freq='m'))
list_months = [i.date() for i in list_months]
df_cpb.columns = list_months
df_cpb = df_cpb.transpose()
cols_cpb = ['importworldipi', 'prodworldipi']
df_cpb.columns = cols_cpb
df_cpb[cols_cpb] = df_cpb[cols_cpb].astype('float')
df_cpb = df_cpb.reset_index().rename(columns={'index': 'date'})
df_cpb['quarter'] = pd.to_datetime(df_cpb['date']).dt.to_period('q')
df_cpb = df_cpb.groupby('quarter')[cols_cpb].mean(numeric_only=True)

# I.C --- Merge base data frames
df = df.merge(df_cpb, how='left', right_index=True, left_index=True)

# II --- Cleaning
# Create max-uncertainty column
df['_zero'] = 0
col_x_cands = []
for i in range(1, 5):
    df['gepu' + str(i)] = df['gepu'].shift(i)
    col_x_cands = col_x_cands + ['gepu' + str(i)]
df['_x'] = df[col_x_cands].max(axis=1)
df['_z'] = 100 * ((df['gepu'] / df['_x']) - 1)
df['maxgepu'] = df[['_zero', '_z']].max(axis=1)
for i in ['_zero', '_x', '_z'] + col_x_cands:
    del df[i]

# III --- Data transformations
# Extract cleaned column names
cols_nice = list(df.columns)
# Split into level and rate columns
cols_levels = ['gdp', 'pc', 'gc', 'gfcf', 'ex', 'im',
               'cpi', 'cpicore', 'cpitransport', 'brent', 'ron95', 'neer', 'reer', 'myrusd',
               'importworldipi', 'prodworldipi', 'gepu']
cols_rates = ['mgs10y', 'klibor3m', 'klibor1m']  # Apply difference, not growth transformation to GEPU
cols_nochange = ['maxgepu']
# QoQ
df_qoq = df.copy()
for i in cols_levels:
    df_qoq[i] = 100 * ((df_qoq[i] / df_qoq[i].shift(1)) - 1)
for i in cols_rates:
    df_qoq[i] = df_qoq[i] - df_qoq[i].shift(1)
# YoY
df_yoy = df.copy()
for i in cols_levels:
    df_yoy[i] = 100 * ((df_yoy[i] / df_yoy[i].shift(4)) - 1)
for i in cols_rates:
    df_yoy[i] = df_yoy[i] - df_yoy[i].shift(4)
# Log-levels
df_ln = df.copy()
for i in cols_levels:
    df_ln[i] = np.log(df[i])
# ln QoQ
df_ln_qoq = df_ln.copy()
for i in cols_levels + cols_rates:
    df_ln_qoq[i] = df_ln_qoq[i] - df_ln_qoq[i].shift(1)
# ln YoY
df_ln_yoy = df_ln.copy()
for i in cols_levels + cols_rates:
    df_ln_yoy[i] = df_ln_yoy[i] - df_ln_yoy[i].shift(4)

# IV --- Pre-export
# Reset index
df = df.reset_index()
df_qoq = df_qoq.reset_index()
df_yoy = df_yoy.reset_index()
df_ln = df_ln.reset_index()
df_ln_qoq = df_ln_qoq.reset_index()
df_ln_yoy = df_ln_yoy.reset_index()
# Convert dates into str
df['quarter'] = df['quarter'].astype('str')
df_qoq['quarter'] = df_qoq['quarter'].astype('str')
df_yoy['quarter'] = df_yoy['quarter'].astype('str')
df_ln['quarter'] = df_ln['quarter'].astype('str')
df_ln_qoq['quarter'] = df_ln_qoq['quarter'].astype('str')
df_ln_yoy['quarter'] = df_ln_yoy['quarter'].astype('str')
# Trim rows with NAs (ignored as we have variables used in subanalysis requiring shorter time series)
# df = df.dropna(axis=0)
# df_qoq = df_qoq.dropna(axis=0)
# df_yoy = df_yoy.dropna(axis=0)
# df_ln = df_ln.dropna(axis=0)
# df_ln_qoq = df_ln_qoq.dropna(axis=0)
# df_ln_yoy = df_ln_yoy.dropna(axis=0)
# V --- Output
df.to_parquet(path_data + 'macro_data_levels.parquet')
df_qoq.to_parquet(path_data + 'macro_data_qoq.parquet')
df_yoy.to_parquet(path_data + 'macro_data_yoy.parquet')
df_ln.to_parquet(path_data + 'macro_data_ln_levels.parquet')
df_ln_qoq.to_parquet(path_data + 'macro_data_ln_qoq.parquet')
df_ln_yoy.to_parquet(path_data + 'macro_data_ln_yoy.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_macro_data: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
