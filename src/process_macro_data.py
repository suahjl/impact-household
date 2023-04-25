import pandas as pd
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

# I --- Load data
# List of ceic series ids and names
seriesids = pd.read_csv(path_data + 'ceic_macro.csv')
# Split into GDP (with historical extension) and others (without historical extension)
seriesids_gdp = seriesids[0:6]
seriesids_others = seriesids[6:]
# Extract only series ID
seriesids_gdp = [re.sub("[^0-9]+", "", i) for i in list(seriesids_gdp['series_id'])]  # keep only numbers
seriesids_gdp = [int(i) for i in seriesids_gdp]
# Extract only series ID (others)
seriesids_others = [re.sub("[^0-9]+", "", i) for i in list(seriesids_others['series_id'])]  # keep only numbers
seriesids_others = [int(i) for i in seriesids_others]
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
# Merge into single data frame
df = df_gdp.merge(df_others, how='outer', right_index=True, left_index=True)
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
    'economic_policy_uncertainty_index_global_ppp_adjusted_gdp': 'gepu',
    'forex_month_average_malaysian_ringgit_to_us_dollar': 'myrusd',
    'government_securities_yield_10_years': 'mgs10y',
    'nominal_effective_exchange_rate_index_bis_2020_100_broad': 'neer',
    'short_term_interest_rate_month_end_klibor_3_months': 'klibor3m'
}
df = df.rename(columns=dict_nice_col_names)

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
cols_levels = ['gdp', 'pc', 'gc', 'gfcf', 'ex', 'im', 'cpi', 'brent', 'neer', 'myrusd']
cols_rates = ['mgs10y', 'klibor3m', 'gepu']  # Apply difference, not growth transformation to GEPU
cols_nochange = ['maxgepu']
# QoQ
df_qoq = df.copy()
for i in cols_levels:
    df_qoq[i] = 100 * ((df_qoq[i] / df_qoq[i].shift(1)) - 1)
for i in cols_rates:
    df_qoq[i] = df_qoq[i] - df_qoq[i].shift(1)
for i in cols_nochange:
    pass
# YoY
df_yoy = df.copy()
for i in cols_levels:
    df_yoy[i] = 100 * ((df_yoy[i] / df_yoy[i].shift(4)) - 1)
for i in cols_rates:
    df_yoy[i] = df_yoy[i] - df_yoy[i].shift(4)
for i in cols_nochange:
    pass

# IV --- Pre-export
# Reset index
df = df.reset_index()
df_qoq = df_qoq.reset_index()
df_yoy = df_yoy.reset_index()
# Convert dates into str
df['quarter'] = df['quarter'].astype('str')
df_qoq['quarter'] = df_qoq['quarter'].astype('str')
df_yoy['quarter'] = df_yoy['quarter'].astype('str')
# Trim rows with NAs
df = df.dropna(axis=0)
df_qoq = df_qoq.dropna(axis=0)
df_yoy = df_yoy.dropna(axis=0)

# V --- Output
df.to_parquet(path_data + 'macro_data_levels.parquet')
df_qoq.to_parquet(path_data + 'macro_data_qoq.parquet')
df_yoy.to_parquet(path_data + 'macro_data_yoy.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_macro_data: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
