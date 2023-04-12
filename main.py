from src.helper import telsendmsg
import os
from dotenv import load_dotenv
import time

time_start = time.time()

load_dotenv()
tel_config = os.getenv('TEL_CONFIG')

# Process raw data
process_raw = False
if process_raw:
    import src.process_raw_2009
    import src.process_raw_2014
    import src.process_raw_2016
    import src.process_raw_2019

process_raw_hhbasis = False
if process_raw_hhbasis:
    import src.process_raw_2014_hhbasis
    import src.process_raw_2016_hhbasis
    import src.process_raw_2019_hhbasis

# Pre-analysis processing
process_pre_analysis = False
if process_pre_analysis:
    import src.process_outliers
    import src.process_consol_group

# Analyse (uses trimmed-outliers per-capita data frames)
generate_adj_analysis = True
if generate_adj_analysis:
    import src.analysis_reg_overall
    import src.analysis_reg_overall_quantile
    time.sleep(15)
    import src.analysis_reg_strat
    time.sleep(30)
    import src.analysis_reg_strat_quantile
    time.sleep(30)
    import src.analysis_reg_strat_subgroups

# Descriptive stats (uses full total-hh-basis data frame)
generate_descriptive = False
if generate_descriptive:
    import src.analysis_descriptive

# Specific analysis
generate_specific = False
if generate_specific:
    import src.analysis_inc_growth_change

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
