import pandas as pd
import telegram_send
from linearmodels import PanelOLS, RandomEffects
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import date
from PIL import Image

plt.switch_backend('agg')


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

    del params_table['SE']

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
    if time_effects and not fixed_effects:
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

    del params_table['SE']

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

    del params_table['SE']

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


def heatmap(input: pd.DataFrame, mask: bool, colourmap: str, outputfile: str, title: str, lb: float, ub: float,
            format: str):
    fig = plt.figure()
    sns.heatmap(input,
                mask=mask,
                annot=True,
                cmap=colourmap,
                center=0,
                annot_kws={'size': 12},
                vmin=lb,
                vmax=ub,
                xticklabels=True,
                yticklabels=True,
                fmt=format)
    plt.title(title, fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close()
    return fig


def pil_img2pdf(list_images: list, extension: str, pdf_name: str):
    seq = list_images.copy()  # deep copy
    list_img = []
    file_pdf = pdf_name + '.pdf'
    run = 0
    for i in seq:
        img = Image.open(i + '.' + extension)
        img = img.convert('RGB')  # PIL cannot save RGBA files as pdf
        if run == 0:
            first_img = img.copy()
        elif run > 0:
            list_img = list_img + [img]
        run += 1
    first_img.save(file_pdf,
                   'PDF',
                   resolution=100.0,
                   save_all=True,
                   append_images=list_img)


def boxplot(
        data: pd.DataFrame,
        y_cols: list,
        x_col: str,
        trace_names: list,
        colours: list,
        main_title: str
):
    # prelims
    d = data.copy()
    # generate figure
    fig = go.Figure()
    # add box plots one by one
    max_candidates = []
    min_candidates = []
    for y, trace_name, colour in zip(y_cols, trace_names, colours):
        fig.add_trace(
            go.Box(
                y=d[y],
                x=d[x_col],
                name=trace_name,
                marker=dict(opacity=0, color=colour),
                boxpoints='outliers'
            )
        )
        max_candidates = max_candidates + [d[y].quantile(q=0.99)]
        min_candidates = min_candidates + [d[y].min()]
    fig.update_yaxes(range=[min(min_candidates), max(max_candidates)],
                     showgrid=True, gridwidth=1, gridcolor='grey')
    # layouts
    fig.update_layout(
        title=main_title,
        plotbg_color='white',
        boxmode='group',
        height=768,
        width=1366
    )
    fig.update_xaxes(categoryorder='category ascending')
    # output
    return fig


def boxplot_time(
        data: pd.DataFrame,
        y_col: str,
        x_col: str,
        t_col: str,
        colours: list,
        main_title: str
):
    # prelims
    d = data.copy()
    list_t = list(d[t_col].unique())
    list_t.sort()
    # generate figure
    fig = go.Figure()
    # add box plots one by one
    max_candidates = []
    min_candidates = []
    for t, colour in zip(list_t, colours):
        fig.add_trace(
            go.Box(
                y=d.loc[d[t_col] == t, y_col],
                x=d.loc[d[t_col] == t, x_col],
                name=str(t),
                marker=dict(opacity=0, color=colour),
                boxpoints='outliers'
            )
        )
        max_candidates = max_candidates + [d.loc[d[t_col] == t, y_col].quantile(q=0.99)]
        min_candidates = min_candidates + [d.loc[d[t_col] == t, y_col].min()]
    fig.update_yaxes(range=[min(min_candidates), max(max_candidates)],
                     showgrid=True, gridwidth=1, gridcolor='grey')
    # layouts
    fig.update_layout(
        title=main_title,
        plot_bgcolor='white',
        boxmode='group',
        font=dict(color='black', size=12),
        height=768,
        width=1366
    )
    fig.update_xaxes(categoryorder='category ascending')
    # output
    return fig
