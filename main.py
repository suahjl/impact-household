from src.helper import telsendmsg
import os
from dotenv import load_dotenv
import time

time_start = time.time()

load_dotenv()
tel_config = os.getenv('TEL_CONFIG')

# Process raw data
process_raw = True
if process_raw:
    import src.process_raw_2009
    import src.process_raw_2014
    import src.process_raw_2016
    import src.process_raw_2019

# Cleaning
import src.process_outliers

# Consolidate into pseudo-panel form
import src.process_consol

# Analyse
import src.analysis_reg_overall

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
