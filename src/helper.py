import pandas as pd
import telegram_send
from linearmodels import PanelOLS, RandomEffects
import statsmodels.formula.api as smf


def telsendimg(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])


def telsendfiles(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])


def telsendmsg(conf='', msg=''):
    telegram_send.send(conf=conf,
                       messages=[msg])


def reg_ols(
        df: pd.DataFrame,
        eqn: str
):
    # Work on copy
    d = df.copy()

    # Estimate model
    mod = smf.ols(formula=eqn, data=d)
    res = mod.fit(cov_type='HC3')
    print(res.summary())

    # Return estimated parameters
    params_table = pd.concat([res.params, res.HC3_se], axis=1)
    params_table.columns = ['Parameter', 'SE']
    params_table['LowerCI'] = params_table['Parameter'] - 1.96 * params_table['SE']
    params_table['UpperCI'] = params_table['Parameter'] + 1.96 * params_table['SE']

    # Return joint test statistics
    joint_teststats = pd.DataFrame(
        {'F-Test': [res.fvalue, res.f_pvalue], }
    )
    joint_teststats = joint_teststats.transpose()
    joint_teststats.columns = ['F', 'P-Value']

    # Regression details
    reg_det = pd.DataFrame(
        {'Observations': [res.nobs],
         'DF Residuals': [res.df_resid]}
    )
    reg_det = reg_det.transpose()
    reg_det.columns = ['Number']

    # Output
    return mod, res, params_table, joint_teststats, reg_det


def fe_reg(
        df: pd.DataFrame,
        y_col: str,
        x_cols: list,
        i_col: str,
        t_col: str,
        fixed_effects: bool,
        time_effects: bool,
        cov_choice: str,
):
    # Work on copy
    d = df.copy()
    d = d.set_index([i_col, t_col])

    # Create eqn
    if not fixed_effects and not time_effects:
        eqn = y_col + '~' + '+'.join(x_cols)
    if fixed_effects and not time_effects:
        eqn = y_col + '~' + '+'.join(x_cols) + '+EntityEffects'
    if time_effects and fixed_effects:
        eqn = y_col + '~' + '+'.join(x_cols) + '+TimeEffects'
    if fixed_effects and time_effects:
        eqn = y_col + '~' + '+'.join(x_cols) + '+EntityEffects+TimeEffects'

    # Estimate model
    mod = PanelOLS.from_formula(formula=eqn, data=d)
    res = mod.fit(cov_type=cov_choice)
    print(res.summary)

    # Return estimated parameters
    params_table = pd.concat([res.params, res.std_errors], axis=1)
    params_table.columns = ['Parameter', 'SE']
    params_table['LowerCI'] = params_table['Parameter'] - 1.96 * params_table['SE']
    params_table['UpperCI'] = params_table['Parameter'] + 1.96 * params_table['SE']

    # Return joint test statistics
    joint_teststats = pd.DataFrame(
        {'F-Test (Poolability)': [res.f_pooled.stat, res.f_pooled.pval],
         'F-Test (Naive)': [res.f_statistic.stat, res.f_statistic.pval],
         'F-Test (Robust)': [res.f_statistic_robust.stat, res.f_statistic_robust.pval]}
    )
    joint_teststats = joint_teststats.transpose()
    joint_teststats.columns = ['F', 'P-Value']

    # Regression details
    reg_det = pd.DataFrame(
        {'Observations': [res.nobs],
         'Entities': [res.entity_info.total],
         'Time Periods': [res.time_info.total]}
    )
    reg_det = reg_det.transpose()
    reg_det.columns = ['Number']

    # Output
    return mod, res, params_table, joint_teststats, reg_det


def re_reg(
        df: pd.DataFrame,
        y_col: str,
        x_cols: list,
        i_col: str,
        t_col: str,
        cov_choice: str,
):
    # Work on copy
    d = df.copy()
    d = d.set_index([i_col, t_col])

    # Create eqn
    eqn = y_col + '~' + '1 +' + '+'.join(x_cols)

    # Estimate model
    mod = RandomEffects.from_formula(formula=eqn, data=d)
    res = mod.fit(cov_type=cov_choice)
    print(res.summary)

    # Return estimated parameters
    params_table = pd.concat([res.params, res.std_errors], axis=1)
    params_table.columns = ['Parameter', 'SE']
    params_table['LowerCI'] = params_table['Parameter'] - 1.96 * params_table['SE']
    params_table['UpperCI'] = params_table['Parameter'] + 1.96 * params_table['SE']

    # Return joint test statistics
    joint_teststats = pd.DataFrame(
        {'F-Test (Naive)': [res.f_statistic.stat, res.f_statistic.pval],
         'F-Test (Robust)': [res.f_statistic_robust.stat, res.f_statistic_robust.pval]}
    )
    joint_teststats = joint_teststats.transpose()
    joint_teststats.columns = ['F', 'P-Value']

    # Regression details
    reg_det = pd.DataFrame(
        {'Observations': [res.nobs],
         'Entities': [res.entity_info.total],
         'Time Periods': [res.time_info.total]}
    )
    reg_det = reg_det.transpose()
    reg_det.columns = ['Number']

    # Output
    return mod, res, params_table, joint_teststats, reg_det
