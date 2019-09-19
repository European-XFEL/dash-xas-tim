"""
Offline data analysis and visualization tool for X-ray Absorption Spectroscopy
(XAS) at European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from flask_caching import Cache
from flask import send_from_directory

from karabo_data import RunDirectory, open_run
from dash_xas_tim import get_run_info, compute_spectrum


ROOT_PATH = "/gpfs/exfel/exp/"
HOME_PATH = osp.expanduser('~')


class Colors:
    FireBrick = '#B22222'
    DodgerBlue = '#1E90FF'
    Chocolate = '#D2691E'
    DarkGrey = '#A9A9A9'
    ForestGreen = '#228B22'
    DarkOrchid = '#9932CC'


MCP_COLORS = (Colors.FireBrick,
              Colors.DodgerBlue,
              Colors.Chocolate,
              Colors.DarkGrey)


app = dash.Dash("XAS TIM Analysis")
app.css.config.serve_locally = False
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 100,
})


for ss in ['base.css', 'style.css']:
    app.css.append_css({"external_url": f"/static/{ss}"})


# -----------------------------
# Convenient functions for Divs
# -----------------------------

def get_property(children, name):
    return children[0]['props'][name]


def get_select_options(name):
    if name.lower() == "topic":
        # folder_list = next(os.walk(osp.join(ROOT_PATH, topic)))[1]
        # print(folder_list
        pass

    return [{'label': 'SCS', 'value': 'scs'}]


def get_selector_cell(name):
    """Get selector cell for the selector bar.

    :param str name: hierarchy name.
    """
    return html.Div(
        className="three-col",
        children=[
            dcc.Dropdown(
                className="run-directory-content",
                options=get_select_options(name),
                placeholder=f"select {name}",
                clearable=False,
            )
        ],
    )


def get_run_directory_select_bar():
    """Get select bar for run directory."""
    return [
        get_selector_cell("Topic"),
        get_selector_cell("Proposal"),
        get_selector_cell("Run"),
    ]


def get_run_directory_input_bar():
    """Get input bar for run directory."""
    return [
        dcc.Input(
            id='run-directory-input',
            className="run-directory-content",
            placeholder='Enter the full path of a run directory ...',
            type='text',
            value=osp.join(HOME_PATH, 'sample_data/xas_tim_p900094_r0348'),
            style={'font-size': '20px'},  # FIXME: CSS does not work
        ),
    ]


def get_ctrl_panel():
    """Return the control panel for parameters tuning."""
    return html.Div(
        children=[
            html.Div(
                className='control-widget',
                children=[
                    html.H6(
                        children="XGM device ID",
                        className="control-label"
                    ),
                    dcc.Dropdown(
                        id='xgm-device-id',
                        className='control-input',
                        options=[],
                        clearable=False,
                    )
                ]
            ),
            html.Div(
                className='control-widget',
                children=[
                    html.H6(
                        children="Softmono device ID",
                        className="control-label"
                    ),
                    dcc.Dropdown(
                        id='mono-device-id',
                        className='control-input',
                        options=[],
                        clearable=False,
                    )
                ]
            ),
            html.Div(
                className='control-widget',
                children=[
                    html.H6(
                        children="Digitizer device ID",
                        className="control-label"
                    ),
                    dcc.Dropdown(
                        id='digitizer-device-id',
                        className='control-input',
                        options=[],
                        clearable=False,
                    )
                ]
            ),
            html.Div(
                className='control-widget row',
                children=[
                    dcc.Input(
                        id='n-pulses-per-train',
                        className='control-input',
                        type='number',
                        min=1,
                        value=1,
                    ),
                    html.H6(
                        children="Number of pulses/train",
                        className="inline-control-label"
                    ),
                ]
            ),
            html.Div(
                className='control-widget row',
                children=[
                    dcc.Input(
                        id='digitizer-stride',
                        className='control-input',
                        type='number',
                        min=1,
                        value=1,
                    ),
                    html.H6(
                        children="Digitizer stride",
                        className="inline-control-label"
                    ),
                ]
            ),
            html.Div(
                className='control-widget row',
                children=[
                    dcc.Input(
                        id='n-energy-bins',
                        className='control-input',
                        type='number',
                        min=10,
                        max=200,
                        value=60
                    ),
                    html.H6(
                        children="Number of energy bins",
                        className="inline-control-label"
                    ),
                ]
            ),
            html.Div(
                className='control-widget row',
                children=[
                    dcc.Input(
                        id='xgm-threshold',
                        className='control-input',
                        type='number',
                        min=0.0,
                        value=0.0,
                    ),
                    html.H6(
                        children='XGM threshold',
                        className="inline-control-label"
                    ),
                ]
            ),
            html.Div(
                className='control-widget row',
                children=[
                    dcc.RadioItems(
                        id='line-plot-mode',
                        className='control-input',
                        options=[
                            {'label': 'lines', 'value': 'lines'},
                            {'label': 'markers', 'value': 'markers'},
                            {'label': 'lines+markers',
                             'value': 'lines+markers'},
                        ],
                        value='lines+markers',
                        labelStyle={'display': 'inline-block'}
                    ),
                    html.H6(
                        children="Line plot mode",
                        className="inline-control-label"
                    ),
                ]
            )
        ]
    )


# -------------------------
# define callback functions
# -------------------------

@app.callback(Output('run-directory-widget', 'children'),
              [Input('run-directory-tab', 'value')])
def render_run_directory_tab(tab):
    if tab == 'gpfs':
        return get_run_directory_select_bar()
    else:
        return get_run_directory_input_bar()


@app.callback([Output('xas-spectrum-graph', 'figure'),
               Output('xgm-spectrum-graph', 'figure')],
              [Input('xgm-device-id', 'value'),
               Input('mono-device-id', 'value'),
               Input('digitizer-device-id', 'value'),
               Input('n-pulses-per-train', 'value'),
               Input('digitizer-stride', 'value'),
               Input('n-energy-bins', 'value'),
               Input('xgm-threshold', 'value'),
               Input('line-plot-mode', 'value')],
              [State('run-directory', 'children')])
def update_graphs(xgm_dev, mono_dev, digitizer_dev, n_pulses,
                  stride, n_bins, xgm_threshold, mode, run_folder):
    try:
        data, spectrum = process_data(
            run_folder, xgm_dev, mono_dev, digitizer_dev, n_pulses, stride,
            n_bins, xgm_threshold)
    except Exception as e:
        print(f"Error in 'update_graph': {repr(e)}")
        raise dash.exceptions.PreventUpdate

    # ------------
    # XAS spectrum
    # ------------

    xas_traces = [
        *[go.Scattergl(x=spectrum['energy'], y=spectrum[f'muA{i}'],
                       marker=dict(color=color, size=10),
                       name=f'MCP{i}',
                       mode=mode)
          for color, i in zip(MCP_COLORS, range(1, 4))],
        go.Bar(
            x=spectrum['energy'],
            y=spectrum['count'],
            marker=dict(color=Colors.DarkGrey, opacity=0.1),
            yaxis='y2',
            name='count'
        )
    ]

    xas_spectrum = {
        'data': xas_traces,
        'layout': {
            'title': '<b>Absorption</b>',
            'xaxis': {
                'title': 'Energy (eV)',
            },
            'yaxis': {
                'title': 'log(-I1/I0)',
            },
            'yaxis2': {
                'title': 'count',
                'overlaying': 'y',
                'side': 'right',
            },
            'legend': {
                'x': 1.05,
                'xanchor': 'left',
                'y': 0.7,
                'yanchor': 'bottom',
            },
            'font': {
                'family': 'Courier New, monospace',
                'size': 16
            },
            'margin': {
                'l': 100, 'b': 50, 't': 50, 'r': 50,
            },
        }
    }

    # ------------
    # XGM spectrum
    # ------------

    xgm_traces = [
        go.Scattergl(
            x=spectrum['energy'], y=spectrum['muXGM'],
            marker=dict(color=Colors.ForestGreen, size=10,
                        line=dict(color=Colors.DarkOrchid, width=2)),
            name='XGM',
            mode=mode),
        go.Bar(
            x=spectrum['energy'],
            y=spectrum['count'],
            marker=dict(color=Colors.DarkGrey, opacity=0.1),
            yaxis='y2',
            name='count',
        )
    ]

    xgm_spectrum = {
        'data': xgm_traces,
        'layout': {
            'xaxis': {
                'title': 'Energy (eV)',
            },
            'yaxis': {
                'title': 'I0 (micro J)',
            },
            'yaxis2': {
                'title': 'count',
                'overlaying': 'y',
                'side': 'right',
            },
            'legend': {
                'x': 1.05,
                'xanchor': 'left',
                'y': 0.7,
                'yanchor': 'bottom',
            },
            'showlegend': True,
            'font': {
                'family': 'Courier New, monospace',
                'size': 16
            },
            'margin': {
                'l': 100, 'b': 50, 't': 50, 'r': 50,
            },
        }
    }

    return xas_spectrum, xgm_spectrum


@app.callback(Output('xgm-mcp-correlation-graph', 'figure'),
              [Input('xgm-device-id', 'value'),
               Input('mono-device-id', 'value'),
               Input('digitizer-device-id', 'value'),
               Input('n-pulses-per-train', 'value'),
               Input('digitizer-stride', 'value'),
               Input('n-energy-bins', 'value'),
               Input('xgm-threshold', 'value'),
               Input('mcp-channel-input', 'value')],
              [State('run-directory', 'children')])
def update_correlation_plot(xgm_dev, mono_dev, digitizer_dev, n_pulses,
                            stride, n_bins, xgm_threshold, mcp_ch, run_folder):
    try:
        data, spectrum = process_data(
            run_folder, xgm_dev, mono_dev, digitizer_dev, n_pulses, stride,
            n_bins, xgm_threshold)
    except Exception as e:
        print(f"Error in 'update_correlation_plot' {repr(e)}")
        raise dash.exceptions.PreventUpdate

    traces = [go.Scattergl(x=data['xgm'], y=data[mcp_ch],
                           marker=dict(color=MCP_COLORS[int(mcp_ch[-1])-1],
                                       opacity=0.2),
                           mode='markers')]
    figure = {
        'data': traces,
        'layout': {
            'xaxis': {
                'title': 'I0 (micro J)',
            },
            'yaxis': {
                'title': 'I1 (arb.)',
            },
            'font': {
                'family': 'Courier New, monospace',
                'size': 16
            },
            'margin': {
                'l': 100, 'b': 50, 't': 50, 'r': 50,
            }
        }
    }

    return figure


@cache.memoize()
def extract_data(run_folder):
    """Extract and cache the useful data for each useful sources."""
    if run_folder is None:
        return

    xgm_srcs = []
    mono_srcs = []
    digitizer_srcs = []
    data = dict()

    try:
        run = RunDirectory(run_folder)

        for src in run.control_sources:
            if re.search(r'XGM/DOOCS', src):
                xgm_srcs.append({'label': src, 'value': src})

                data[src] = dict()
                data[src]['output'] = run.get_array(f'{src}:output',
                                                    'data.intensitySa3TD')
                data[src]['photonFlux'] = run.get_dataframe(
                    fields=[(src, 'pulseEnergy.photonFlux.value')])

            elif re.search(r'MONO.*PHOTON_ENERGY', src):
                mono_srcs.append({'label': src, 'value': src})

                data[src] = dict()
                data[src]['actualEnergy'] = run.get_dataframe(
                    fields=[(src, 'actualEnergy.value')])

            elif re.search(r'ADQ/ADC', src):
                digitizer_srcs.append({'label': src, 'value': src})

                data[src] = dict()
                # MCP1 -> 'D', MCP2 -> 'C', MCP3 -> 'B', MCP4 -> 'A'
                mcp_chs = {
                    f'mcp{i}': {
                        'apd': f"digitizers.channel_1_{ch}.apd.pulseIntegral"}
                    for i, ch in enumerate(('D', 'C', 'B', 'A'), 1)
                }
                data[src]['network'] = dict()
                for ch, v in mcp_chs.items():
                    data[src]['network'][ch] = run.get_array(
                        f"{src}:network", v['apd'])

    except Exception as e:
        print(f"Failed to load a run directory: {repr(e)}")

    return data, xgm_srcs, mono_srcs, digitizer_srcs


@cache.memoize()
def process_data(run_folder, xgm_dev, mono_dev, digitizer_dev,
                 n_pulses, stride, n_bins, xgm_threshold):
    cache.delete_memoized(extract_data)
    data, xgm_srcs, mono_srcs, digitizer_srcs = extract_data(run_folder)

    xgm_arr = data[xgm_dev]['output']

    mono_df = data[mono_dev]['actualEnergy']
    mono_df.rename(columns=lambda x: x.split('/')[-1], inplace=True)

    digitizer_arrs = data[digitizer_dev]['network']

    processed = {
        "xgm": xgm_arr.values[..., 0:n_pulses].flatten(),
        'energy': np.repeat(mono_df['actualEnergy'], n_pulses),
    }

    # TODO: add check for train IDs

    for k, v in digitizer_arrs.items():
        # the sign of I1 is flipped
        processed[k] = -v.values[..., :n_pulses*stride:stride].flatten()

    min_len = np.inf
    max_len = 1
    for key in processed:
        l = len(processed[key])
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l
    if min_len != max_len:
        for key in processed:
            processed[key] = processed[key][:min_len]

    data_df = pd.DataFrame(processed)

    data_df.query(f"xgm >= {xgm_threshold}", inplace=True)

    spectrum = compute_spectrum(data_df, n_bins=n_bins)

    return data_df, spectrum


@app.callback([Output('run-directory', 'children'),
               Output('xgm-device-id', 'options'),
               Output('xgm-device-id', 'value'),
               Output('mono-device-id', 'options'),
               Output('mono-device-id', 'value'),
               Output('digitizer-device-id', 'options'),
               Output('digitizer-device-id', 'value')],
              [Input('load-run-button', 'n_clicks')],
              [State('run-directory-tab', 'value'),
               State('run-directory-widget', 'children')])
def load_run_directory(n_clicks, input_type, widget):
    if widget is None:
        raise dash.exceptions.PreventUpdate

    if input_type == 'local':
        run_folder = osp.join(HOME_PATH, get_property(widget, 'value'))
    else:
        raise dash.exceptions.PreventUpdate

    data, xgm_srcs, mono_srcs, digitizer_srcs = extract_data(run_folder)

    return (run_folder,
            xgm_srcs, xgm_srcs[0]['value'],
            mono_srcs, mono_srcs[0]['value'],
            digitizer_srcs, digitizer_srcs[0]['value'])


# define content and layout of the web page
app.layout = html.Div(children=[
    html.Div(
        children=[
            html.H3(
                "X-ray Absorption Spectroscopy (XAS) with TIM",
            ),
        ]
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="run-directory-bar",
                children=[
                    dcc.Tabs(
                        id="run-directory-tab",
                        value='local',
                        children=[
                            dcc.Tab(label='GPFS', value='gpfs'),
                            dcc.Tab(label='Local', value='local'),
                        ],
                    ),
                    html.Div(
                        id='run-directory-widget',
                    ),
                    html.Div(
                        id="button-top-bar",
                        className="row",
                        children=[
                            html.Button(
                                'Load run',
                                id='load-run-button',
                                className='button top-bar-button',
                                type='submit',
                            ),
                        ]
                    ),
                    html.Div(id="run-directory", className='display-none')
                ]
            ),
        ]
    ),
    html.Div(
        className="content-row",
        children=[
            html.Div(
                id='spectrum-graph',
                className='row display-inlineblock',
                children=[
                    dcc.Graph(
                        id='xas-spectrum-graph',
                        className='graph',
                        animate=False
                    ),
                    # animation is not supported for bar chart yet
                    dcc.Graph(
                        id='xgm-spectrum-graph',
                        className='graph',
                        animate=False
                    ),
                ],
            ),
            html.Div(
                id='control-panel',
                className='row',
                children=get_ctrl_panel(),
            ),
            html.Div(
                id='correlation-graph',
                className='row display-inlineblock',
                children=[
                    dcc.Dropdown(
                        id="mcp-channel-input",
                        options=[{'label': f'MCP{i}', 'value': f'mcp{i}'}
                                 for i in range(1, 5)],
                        value='mcp1',
                        clearable=False,
                    ),
                    dcc.Graph(id='xgm-mcp-correlation-graph'),
                ]
            ),
        ],
    ),
])


# -----
# Flask
# -----

@app.server.route('/static/<stylesheet>')
def serve_stylesheet(stylesheet):
    path = osp.join(osp.dirname(osp.abspath(__file__)), 'assets')
    return send_from_directory(path, stylesheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="run Dash in debug mode",
                        action="store_true")
    args = parser.parse_args()

    # The default port for DASH app is 8050. We fix the port here to avoid
    # multiple apps running in a same machine. If there is already one
    # running, all users can access it.
    app.run_server(debug=args.debug, host='0.0.0.0', port=8051, processes=1)
