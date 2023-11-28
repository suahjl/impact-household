# %%
from helper import telsendmsg
import os
from dotenv import load_dotenv
import time

time_start = time.time()

# %%
load_dotenv()
tel_config = os.getenv('TEL_CONFIG')

# %%
# Process raw data
process_raw = False
if process_raw:
    import process_raw_2009
    import process_raw_2014
    import process_raw_2016
    import process_raw_2019
    import process_raw_2022
    import process_raw_2022_bnm

process_raw_hhbasis = False
if process_raw_hhbasis:
    import process_raw_2014_hhbasis
    import process_raw_2016_hhbasis
    import process_raw_2019_hhbasis
    import process_raw_2022_hhbasis
    import process_raw_2022_bnm_hhbasis

process_raw_equivalised = False
if process_raw_equivalised:
    import process_raw_2014_equivalised
    import process_raw_2016_equivalised
    import process_raw_2019_equivalised
    import process_raw_2022_equivalised
    import process_raw_2022_bnm_equivalised

# %%
# Pre-analysis processing
process_pre_analysis = False
if process_pre_analysis:
    import process_outliers
    import process_consol_group

# %%
# Analyse (uses trimmed-outliers hhbasis data frames)
generate_adj_analysis = True
if generate_adj_analysis:
    import analysis_reg_overall
    import analysis_reg_overall_quantile

    # time.sleep(15)
    import analysis_reg_strat

    # time.sleep(45)
    import analysis_reg_strat_quantile

    # time.sleep(30)
    import analysis_reg_strat_subgroups

# %%
# Descriptive stats
generate_descriptive = True
if generate_descriptive:
    import analysis_descriptive
    import analysis_descriptive_inc_allbasis_fixedincgroups
    # import analysis_inc_growth_change

# %%
# Macro analysis
generate_macro_analysis = False
if generate_macro_analysis:
    import analysis_macro_var_varyx
    import analysis_macro_var_elec_varyx
    import analysis_macro_var_petrol_varyx

# %%
# Policy simulation
generate_sim = False
if generate_sim:
    import analysis_descriptive_cost_matrix
    import analysis_impact_sim
    import analysis_impact_sim_elec
    import analysis_impact_sim_petrol
    import analysis_impact_sim_net

# %%
# Notify
telsendmsg(
    conf=tel_config,
    msg='impact-household --- main: COMPLETED'
)

# End
time_check_text = '\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----'
print(time_check_text)
telsendmsg(
    conf=tel_config,
    msg=time_check_text
)
