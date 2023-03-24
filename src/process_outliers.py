# Process outliers of respective raw files, before consolidating for further analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
path_2009 = './data/hies_2009/'
path_2014 = './data/hies_2014/'
path_2016 = './data/hies_2016/'
path_2019 = './data/hies_2019/'

# I --- Load processed vintages
df_09 = pd.read_parquet(path_2009 + 'hies_2009_consol.parquet')
df_14 = pd.read_parquet(path_2014 + 'hies_2014_consol.parquet')
df_16 = pd.read_parquet(path_2016 + 'hies_2016_consol.parquet')
df_19 = pd.read_parquet(path_2019 + 'hies_2019_consol.parquet')


# II --- Outliers by income and spending (items 1 - 12)

def outlier_kmeans(data, cols_y_x, threshold):
    # Prelims
    d = data.copy()
    # Model
    dist = KMeans(n_clusters=5, init='random', n_init='auto', random_state=11353415, copy_x=True, algorithm='lloyd') \
        .fit_transform(d[cols_y_x])
    dist = pd.DataFrame(dist)
    # Narrow down to own-cluster distance
    dist['own_cluster_dist'] = pd.Series(dist.min(axis=1))
    # Use threshold to find out if an observation is an 'outlier'
    dist.loc[dist['own_cluster_dist'] >= dist['own_cluster_dist'].quantile(q=threshold), 'outlier'] = 1
    dist.loc[dist['outlier'].isna(), 'outlier'] = 0
    dist = dist[['own_cluster_dist', 'outlier']]
    # Merge back with main data
    d = d.merge(dist, right_index=True, left_index=True)
    # Drop outliers
    d = d[d['outlier'] == 0].copy()
    # Trim columns
    for i in ['own_cluster_dist', 'outlier']:
        del d[i]
    # Output
    return d


df_19 = outlier_kmeans(data=df_19, cols_y_x=['gross_income', 'cons_01_12'], threshold=0.95)
df_16 = outlier_kmeans(data=df_16, cols_y_x=['gross_income', 'cons_01_12'], threshold=0.95)
df_14 = outlier_kmeans(data=df_14, cols_y_x=['gross_income', 'cons_01_12'], threshold=0.95)
df_09 = outlier_kmeans(data=df_09, cols_y_x=['gross_income', 'cons_01_12'], threshold=0.95)

# X --- Output
df_19.to_parquet(path_2019 + 'hies_2019_consol_trimmedoutliers.parquet')
df_16.to_parquet(path_2016 + 'hies_2016_consol_trimmedoutliers.parquet')
df_14.to_parquet(path_2014 + 'hies_2014_consol_trimmedoutliers.parquet')
df_09.to_parquet(path_2009 + 'hies_2009_consol_trimmedoutliers.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_outliers: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
