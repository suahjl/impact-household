import pandas as pd
import telegram_send
from linearmodels import PanelOLS, RandomEffects
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import date
from PIL import Image
from ceic_api_client.pyceic import Ceic
import re
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
plt.switch_backend('agg')


# --- Notifications

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


# --- Data
def get_data_from_api_ceic(
        series_ids: list[float], series_names: list[str], start_date: date, historical_extension: bool = False
) -> pd.DataFrame:
    """
    Get CEIC data.
    Receive a list of series IDs (e.g., [408955597] for CPI inflation YoY) from CEIC
    and output a pandas data frame (for single entity time series).

    :series_ids `list[float]`: a list of CEIC Series IDs\n
    :start_date `date`: a date() object of the start date e.g. date(1991, 1, 1)\n
    "continuous `Optional[boolean]`: When set to true, series will include extended historical timepoints\n
    :return `pd.DataFrame`: A DataFrame instance of the data
    """
    df = pd.DataFrame()
    series_list = ",".join(map(str, series_ids))

    if historical_extension == False:
        PATH = f"https://api.ceicdata.com/v2//series/{series_list}/data?format=json&start_date={start_date}"
        response = requests.get(f"{PATH}&token={os.getenv('CEIC_API_KEY')}")
        content = json.loads(response.text)["data"]
    else:
        content = []
        for series in series_ids:
            PATH = f"https://api.ceicdata.com/v2//series/{series}/data?format=json&start_date={start_date}"
            response = requests.get(
                f"{PATH}&with_historical_extension=True&token={os.getenv('CEIC_API_KEY')}"
            )
            content = content + json.loads(response.text)["data"]
    for i, j in zip(range(len(series_ids)), series_names):  # series names not in API json
        data = pd.DataFrame(content[i]["timePoints"])[["date", "value"]]
        # name = content[i]["layout"][0]["table"]["name"]
        name = "_".join(j.split(": ")[1:])  # name --> j
        data["name"] = re.sub("[^A-Za-z0-9]+", "_", name).lower()
        country = content[i]["layout"][0]["topic"]["name"]  # section --> topic
        data["country"] = re.sub("[^A-Za-z0-9]+", "_", country).lower()
        df = pd.concat([df, data])

    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    return df


def get_data_from_ceic(
        series_ids: list[float], start_date: date, historical_extension: bool = False
) -> pd.DataFrame:
    """
    Get CEIC data.
    Receive a list of series IDs (e.g., [408955597] for CPI inflation YoY) from CEIC
    and output a pandas data frame (for single entity time series).

    :series_ids `list[float]`: a list of CEIC Series IDs\n
    :start_date `date`: a date() object of the start date e.g. date(1991, 1, 1)\n
    :return `pd.DataFrame`: A DataFrame instance of the data
    """
    Ceic.login(username=os.getenv("CEIC_USERNAME"), password=os.getenv("CEIC_PASSWORD"))

    df = pd.DataFrame()
    content = []
    if historical_extension == False:
        content = Ceic.series(series_id=series_ids, start_date=start_date).data
    else:
        for series in series_ids:
            content += Ceic.series(
                series_id=series,
                start_date=start_date,
                with_historical_extension=True,
            ).data

    for i in range(len(series_ids)):  # for i in range(len(content))
        data = pd.DataFrame(
            [(tp._date, tp.value) for tp in content[i].time_points],
            columns=["date", "value"],
        )
        data["name"] = re.sub("[^A-Za-z0-9]+", "_", content[i].metadata.name).lower()
        data["country"] = re.sub(
            "[^A-Za-z0-9]+", "_", content[i].metadata.country.name
        ).lower()
        df = pd.concat([df, data])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    return df


# --- Linear regressions

def reg_ols(
        df: pd.DataFrame,
        eqn: str
):
    # Work on copy
    d = df.copy()

    # Estimate model
    mod = smf.ols(formula=eqn, data=d)
    res = mod.fit(cov_type='HC3')
    # print(res.summary())

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
    # print(res.summary)

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
    # print(res.summary)

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


# --- TIME SERIES MODELS


def est_varx(
        df: pd.DataFrame,
        cols_endog: list,
        run_varx: bool,
        cols_exog: list,
        choice_ic: str,
        choice_trend: str,
        choice_horizon: int,
        choice_maxlags: int
):
    # Work on copy
    d = df.copy()

    # Estimate model
    if run_varx:
        mod = smt.VAR(endog=d[cols_endog], exog=d[cols_exog])
    if not run_varx:
        mod = smt.VAR(endog=d[cols_endog])
    res = mod.fit(ic=choice_ic, trend=choice_trend, maxlags=choice_maxlags)
    irf = res.irf(periods=choice_horizon)

    # Output
    return res, irf


# --- CHARTS

def heatmap(input: pd.DataFrame,
            mask: bool,
            colourmap: str,
            outputfile: str,
            title: str,
            lb: float,
            ub: float,
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


def barchart(
        data: pd.DataFrame,
        y_col: str,
        x_col: str,
        main_title: str,
        decimal_points: int
):
    # generate figure
    fig = go.Figure()
    # add bar chart
    fig.add_trace(
        go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker=dict(color='lightblue'),
            text=data[y_col].round(decimal_points).astype('str'),
            textposition='outside'
        )
    )
    # layouts
    fig.update_layout(
        title=main_title,
        plot_bgcolor='white',
        font=dict(color='black', size=16),
        height=768,
        width=1366,
    )
    fig.update_traces(textfont_size=28)
    # output
    return fig


def manual_irf_subplots(
        data,
        endog,
        shock_col,
        response_col,
        irf_col,
        horizon_col,
        main_title,
        maxrows,
        maxcols,
        line_colour,
        annot_size,
        font_size
):
    # Create titles first
    titles = []
    for response in endog:
        for shock in endog:
            titles = titles + [shock + ' -> ' + response]
    maxr = maxrows
    maxc = maxcols
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=titles)
    nr = 1
    nc = 1
    # columns: shocks, rows: responses; move columns, then rows
    for response in endog:
        for shock in endog:
            # Data copy
            d = data[(data[shock_col] == shock) & (data[response_col] == response)].copy()
            # Add selected series
            fig.add_trace(
                go.Scatter(
                    x=d[horizon_col].astype('str'),
                    y=d[irf_col],
                    mode='lines',
                    line=dict(width=3, color=line_colour)
                ),
                row=nr,
                col=nc
            )
            # Add zero line
            fig.add_hline(
                y=0,
                line_width=1,
                line_dash='solid',
                line_color='grey',
                row=nr,
                col=nc
            )
            # Move to next subplot
            nc += 1
            if nr > maxr:
                raise NotImplementedError('More subplots than allowed by dimension of main plot!')
            if nc > maxc:
                nr += 1  # next row
                nc = 1  # reset column
    for annot in fig['layout']['annotations']:
        annot['font'] = dict(size=annot_size, color='black')  # subplot title font size
    fig.update_layout(
        title=main_title,
        # yaxis_title=y_title,
        plot_bgcolor='white',
        hovermode='x',
        font=dict(color='black', size=font_size),
        showlegend=False,
        height=768,
        width=1366
    )
    # output
    return fig
