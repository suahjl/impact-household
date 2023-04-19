# Process outliers of respective raw files, before consolidating for further analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
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
opt_random_state = 11353415

# I --- Load processed vintages
df_09 = pd.read_parquet(path_2009 + 'hies_2009_consol.parquet')
df_14 = pd.read_parquet(path_2014 + 'hies_2014_consol.parquet')
df_16 = pd.read_parquet(path_2016 + 'hies_2016_consol.parquet')
df_19 = pd.read_parquet(path_2019 + 'hies_2019_consol.parquet')

df_14_hhbasis = pd.read_parquet(path_2014 + 'hies_2014_consol_hhbasis.parquet')
df_16_hhbasis = pd.read_parquet(path_2016 + 'hies_2016_consol_hhbasis.parquet')
df_19_hhbasis = pd.read_parquet(path_2019 + 'hies_2019_consol_hhbasis.parquet')

df_14_equivalised = pd.read_parquet(path_2014 + 'hies_2014_consol_equivalised.parquet')
df_16_equivalised = pd.read_parquet(path_2016 + 'hies_2016_consol_equivalised.parquet')
df_19_equivalised = pd.read_parquet(path_2019 + 'hies_2019_consol_equivalised.parquet')

# II --- Outliers by income and spending (items 1 - 12)
# Define functions
def outlier_kmeans(data, cols_y_x, threshold):
    # Prelims
    d = data.copy()
    # Model
    dist = KMeans(
        n_clusters=5,
        init='random',
        n_init='auto',
        random_state=opt_random_state,
        copy_x=True,
        algorithm='lloyd'
    ) \
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
    # Tabulate outliers
    print(tabulate(pd.DataFrame(d['outlier'].value_counts()), tablefmt='psql', headers='keys', showindex='always'))
    # Drop outliers
    d = d[d['outlier'] == 0].copy()
    # Trim columns
    for i in ['own_cluster_dist', 'outlier']:
        del d[i]
    # Output
    return d


def outlier_isolationforest(data, cols_x, opt_max_samples, opt_threshold):
    # Prelims
    d = data.copy()
    # Model
    res = IsolationForest(
        random_state=opt_random_state,
        max_samples=opt_max_samples,
        n_jobs=7,
        contamination=opt_threshold
    ) \
        .fit_predict(d[cols_x])
    res = pd.DataFrame(res, columns=['outlier'])
    # Merge with original
    d = d.merge(res, right_index=True, left_index=True)
    # Tabulate outliers
    print(tabulate(pd.DataFrame(d['outlier'].value_counts()), tablefmt='psql', headers='keys', showindex='always'))
    # Drop outliers
    d = d[d['outlier'] == 1].copy()  # outliers = -1
    # Trim columns
    for i in ['outlier']:
        del d[i]
    # Output
    return d


# detect and drop outliers
cols_features_base = ['gross_income'] + \
                     ['salaried_wages', 'other_wages', 'asset_income', 'gross_transfers'] + \
                     ['cons_01_12', 'cons_01_13'] + \
                     ['cons_0' + str(i) for i in range(1, 10)] + \
                     ['cons_1' + str(i) for i in range(0, 4)]
cols_features = cols_features_base + ['cons_0722_fuel', 'cons_07_ex_bigticket']

use_iforest = True
if use_iforest:
    df_19 = outlier_isolationforest(data=df_19, cols_x=cols_features,
                                    opt_max_samples=int(len(df_19) / 100), opt_threshold=0.01)
    df_16 = outlier_isolationforest(data=df_16, cols_x=cols_features,
                                    opt_max_samples=int(len(df_16) / 100), opt_threshold=0.01)
    df_14 = outlier_isolationforest(data=df_14, cols_x=cols_features,
                                    opt_max_samples=int(len(df_14) / 100), opt_threshold=0.01)
    df_09 = outlier_isolationforest(data=df_09, cols_x=cols_features_base,
                                    opt_max_samples=int(len(df_09) / 100), opt_threshold=0.01)

    df_19_hhbasis = outlier_isolationforest(data=df_19_hhbasis, cols_x=cols_features,
                                            opt_max_samples=int(len(df_19) / 100), opt_threshold=0.01)
    df_16_hhbasis = outlier_isolationforest(data=df_16_hhbasis, cols_x=cols_features,
                                            opt_max_samples=int(len(df_16) / 100), opt_threshold=0.01)
    df_14_hhbasis = outlier_isolationforest(data=df_14_hhbasis, cols_x=cols_features,
                                            opt_max_samples=int(len(df_14) / 100), opt_threshold=0.01)

use_kmeans = False
if use_kmeans:
    df_19 = outlier_kmeans(data=df_19, cols_y_x=cols_features, threshold=0.99)
    df_16 = outlier_kmeans(data=df_16, cols_y_x=cols_features, threshold=0.99)
    df_14 = outlier_kmeans(data=df_14, cols_y_x=cols_features, threshold=0.99)
    df_09 = outlier_kmeans(data=df_09, cols_y_x=cols_features_base, threshold=0.99)

    df_19_hhbasis = outlier_kmeans(data=df_19_hhbasis, cols_y_x=cols_features, threshold=0.99)
    df_16_hhbasis = outlier_kmeans(data=df_16_hhbasis, cols_y_x=cols_features, threshold=0.99)
    df_14_hhbasis = outlier_kmeans(data=df_14_hhbasis, cols_y_x=cols_features, threshold=0.99)

    df_19_equivalised = outlier_kmeans(data=df_19_equivalised, cols_y_x=cols_features, threshold=0.99)
    df_16_equivalised = outlier_kmeans(data=df_16_equivalised, cols_y_x=cols_features, threshold=0.99)
    df_14_equivalised = outlier_kmeans(data=df_14_equivalised, cols_y_x=cols_features, threshold=0.99)

# X --- Output
df_19.to_parquet(path_2019 + 'hies_2019_consol_trimmedoutliers.parquet')
df_16.to_parquet(path_2016 + 'hies_2016_consol_trimmedoutliers.parquet')
df_14.to_parquet(path_2014 + 'hies_2014_consol_trimmedoutliers.parquet')
df_09.to_parquet(path_2009 + 'hies_2009_consol_trimmedoutliers.parquet')

df_19_hhbasis.to_parquet(path_2019 + 'hies_2019_consol_hhbasis_trimmedoutliers.parquet')
df_16_hhbasis.to_parquet(path_2016 + 'hies_2016_consol_hhbasis_trimmedoutliers.parquet')
df_14_hhbasis.to_parquet(path_2014 + 'hies_2014_consol_hhbasis_trimmedoutliers.parquet')

df_19_equivalised.to_parquet(path_2019 + 'hies_2019_consol_equivalised_trimmedoutliers.parquet')
df_16_equivalised.to_parquet(path_2016 + 'hies_2016_consol_equivalised_trimmedoutliers.parquet')
df_14_equivalised.to_parquet(path_2014 + 'hies_2014_consol_equivalised_trimmedoutliers.parquet')

# X --- Notify
telsendmsg(conf=tel_config,
           msg='impact-household --- process_outliers: COMPLETED')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
