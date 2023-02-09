import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

import explanations

NETCOM_LOGO = (
    "https://www.arcyber.army.mil/portals/78/website-assets/images/units/netcom-min.png"
)


def layout(DASHBOARD_DATA):
    navbar = dbc.Navbar(
        html.Div(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=NETCOM_LOGO, height="30px")),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "NETCOM Outlier Detection Dashboard",
                                    className="ms-2",
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="127.0.0.1:8051",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            ]
        ),
        color="dark",
        dark=True,
    )
    fig = px.scatter(
        DASHBOARD_DATA.graph_data,
        x="x",
        y="y",
        color="score",
        color_continuous_scale="Portland",
    )
    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader("Top 10 feature importance"),
                    dbc.ModalBody(
                        [
                            html.H4(id="hover_info"),
                            dcc.Graph(id="modal_graph", figure=fig),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ml-auto")
                    ),
                ],
                id="modal",
                centered=True,
            ),
        ]
    )

    body = html.Div(
        children=[
            # header
            html.Br(),
            # first row
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                "Number of Anomaly Groups: ",
                                dcc.Dropdown(
                                    options=[
                                        {"label": i, "value": i}
                                        for i in DASHBOARD_DATA.feature_columns[2:]
                                    ],
                                    value=str(DASHBOARD_DATA.optimal_k) + " clusters",
                                    id="dropdown",
                                    style={
                                        "width": "50%",
                                        "offset": 1,
                                    },
                                    clearable=False,
                                ),
                            ]
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                "Cluster Number: ",
                                dcc.Dropdown(
                                    options=[
                                        {"label": i, "value": i}
                                        for i in DASHBOARD_DATA.cluster_options[-1]
                                    ],
                                    value="cluster0",
                                    id="cluster_dropdown",
                                    style={
                                        "width": "60%",
                                        "offset": 1,
                                    },
                                    clearable=False,
                                ),
                            ]
                        ),
                        width=7,
                    ),
                ],
                style={
                    "background-color": "white",
                    "border-radius": "15px",
                    "padding": "5px",
                    "height": "10%",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.I(
                                children=" ",
                                style={
                                    "color": "black",
                                    "text-align": "center",
                                    "font-size": "20px",
                                    "font-weight": "Normal",
                                },
                                className="fas fa-cog",
                                id="target_scatter_cog",
                            ),
                            html.I(
                                children="Scatter Plot",
                                style={
                                    "color": "black",
                                    "text-align": "center",
                                    "font-size": "20px",
                                    "font-weight": "Normal",
                                },
                                className="fas fa-question-circle fa-lg",
                                id="target_scatter",
                            ),
                            dbc.Tooltip(
                                explanations.CLUSTER_EXPLAIN_HOW,
                                target="target_scatter",
                                placement="top-start",
                            ),
                            dbc.Tooltip(
                                explanations.CLUSTER_EXPLAIN_WHAT,
                                target="target_scatter_cog",
                                placement="top-start",
                            ),
                            dcc.Graph(id="scatter"),
                        ],
                        width=5,
                    ),
                    dbc.Col(
                        [
                            html.I(
                                children=" ",
                                style={
                                    "color": "black",
                                    "text-align": "center",
                                    "font-size": "20px",
                                    "font-weight": "Normal",
                                },
                                className="fas fa-cog",
                                id="target_rule_cog",
                            ),
                            html.I(
                                children="Rule Mining",
                                style={
                                    "color": "black",
                                    "text-align": "center",
                                    "font-size": "20px",
                                    "font-weight": "Normal",
                                },
                                className="fas fa-question-circle fa-lg",
                                id="target_rule",
                            ),
                            dbc.Tooltip(
                                explanations.RULE_MINING_HOW,
                                target="target_rule",
                                placement="top-start",
                            ),
                            dbc.Tooltip(
                                explanations.RULE_MINING_WHAT,
                                target="target_rule_cog",
                                placement="top-start",
                            ),
                            html.Div("Type in your threshold:"),
                            dcc.Input(
                                id="mass_input",
                                type="number",
                                placeholder="COVERAGE",
                            ),
                            dcc.Input(
                                id="purity_input",
                                type="number",
                                placeholder="PURITY",
                            ),
                            html.Button("Show Rules", id="submit-rules", n_clicks=0),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            "Rule Candidates",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(id="rules", children=[]),
                                    ]
                                )
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.I(
                                        children=" ",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-cog",
                                        id="target_score_cog",
                                    ),
                                    html.I(
                                        children="Rule Design Interface",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-question-circle fa-lg",
                                        id="target_score",
                                    ),
                                    dbc.Tooltip(
                                        explanations.RULE_MINING_HOW,
                                        target="target_score",
                                        placement="top-start",
                                    ),
                                    dbc.Tooltip(
                                        explanations.RULE_MINING_WHAT,
                                        target="target_score_cog",
                                        placement="top-start",
                                    ),
                                ]
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            "Selected Rules",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(id="output-container-range-slider"),
                                        html.Div(id="selected_rule", children=[]),
                                    ]
                                )
                            ),
                            html.Div(
                                [
                                    "Feature Number: ",
                                    dcc.Dropdown(
                                        options=[
                                            {"label": i, "value": i}
                                            for i in DASHBOARD_DATA.dropdown_options
                                        ],
                                        value=DASHBOARD_DATA.dropdown_options[-1],
                                        id="feature_dropdown",
                                        style={
                                            "width": "50%",
                                            "offset": 1,
                                        },
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Button("Add Predicates", id="submit-val", n_clicks=0),
                            html.Div(id="rules_slider", children=[]),
                            html.Button(
                                "Calculate Scores", id="score_button", n_clicks=0),
                            html.Div(id="scores", children=[]),
                            html.Button("Save rules", id= "save_rules", n_clicks = 0),
                            html.Div(id="saved", children = [])
                        ],
                    ),
                ],
                justify="center",
                style={
                    "background-color": "white",
                    "border-radius": "15px",
                    "padding": "5px",
                    "height": "5%",
                },
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.I(
                                        children=" ",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-cog",
                                        id="target_histogram_cog",
                                    ),
                                    html.I(
                                        children="Histogram",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-question-circle fa-lg",
                                        id="target_histogram",
                                    ),
                                    dbc.Tooltip(
                                        explanations.HISTOGRAM_EXPLAIN_WHAT,
                                        target="target_histogram",
                                        placement="top-start",
                                    ),
                                    dbc.Tooltip(
                                        explanations.HISTOGRAM_EXPLAIN_HOW,
                                        target="target_histogram_cog",
                                        placement="top-start",
                                    ),
                                    dcc.Dropdown(
                                        options=[
                                            {"label": i, "value": i}
                                            for i in DASHBOARD_DATA.dropdown_options
                                        ],
                                        value=DASHBOARD_DATA.dropdown_options[-1],
                                        id="dropdown_histogram",
                                        clearable=False,
                                        style={
                                            "width": "70%",
                                            "offset": 1,
                                        },
                                    ),
                                    dcc.Graph(id="histogram"),
                                ],
                                className="p-5 text-muted",
                                style={
                                    "background-color": "white",
                                    "border-radius": "15px",
                                    "padding": "10px",
                                },
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.I(
                                        children=" ",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-cog",
                                        id="target_density_cog",
                                    ),
                                    html.I(
                                        children="Density Plot",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-question-circle fa-lg",
                                        id="target_density",
                                    ),
                                    dbc.Tooltip(
                                        explanations.DENSITY_EXPLAIN_WHAT,
                                        target="target_density",
                                        placement="top-start",
                                    ),
                                    dbc.Tooltip(
                                        explanations.DENSITY_EXPLAIN_HOW,
                                        target="target_density_cog",
                                        placement="top-start",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Dropdown(
                                                        options=[
                                                            {"label": i, "value": i}
                                                            for i in DASHBOARD_DATA.dropdown_options
                                                        ],
                                                        value=DASHBOARD_DATA.dropdown_options[
                                                            -1
                                                        ],
                                                        id="dropdown_density1",
                                                        clearable=False,
                                                    )
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    dcc.Dropdown(
                                                        options=[
                                                            {"label": i, "value": i}
                                                            for i in DASHBOARD_DATA.dropdown_options
                                                        ],
                                                        value=DASHBOARD_DATA.dropdown_options[
                                                            -2
                                                        ],
                                                        id="dropdown_density2",
                                                        clearable=False,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    dcc.Graph(id="density"),
                                ],
                                className="p-5 text-muted",
                                style={
                                    "background-color": "white",
                                    "border-radius": "15px",
                                    "padding": "10px",
                                },
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.I(
                                        children=" ",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                        },
                                        className="fas fa-cog",
                                        id="target_parallel_cog",
                                    ),
                                    html.I(
                                        children="Parallel Plot",
                                        style={
                                            "color": "black",
                                            "text-align": "center",
                                            "font-size": "20px",
                                            "font-weight": "Normal",
                                            "padding-bottom": "20px",
                                        },
                                        className="fas fa-question-circle fa-lg",
                                        id="target_parallel",
                                    ),
                                    dbc.Tooltip(
                                        explanations.PARALLEL_EXPLAIN_WHAT,
                                        target="target_parallel",
                                        placement="top-start",
                                    ),
                                    dbc.Tooltip(
                                        explanations.PARALLEL_EXPLAIN_HOW,
                                        target="target_parallel_cog",
                                        placement="top-start",
                                    ),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="parallel"), width=8),
                                            dbc.Col(
                                                [
                                                    html.Div(
                                                        [
                                                            "Show features in parallel plot",
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                list(range(3, 8)),
                                                                3,
                                                                id="parallel_feature_num",
                                                                style={
                                                                    "display": "inline-block"
                                                                },
                                                            )
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    html.Div(
                                                        id="parallel_features",
                                                        children=[],
                                                    ),
                                                ],
                                                width=4,
                                            ),
                                        ],
                                        align="start",
                                    ),
                                ],
                                className="p-5 text-muted",
                                style={
                                    "background-color": "white",
                                    "border-radius": "15px",
                                    "padding": "5px",
                                },
                            )
                        ],
                        width=4,
                    ),
                ],
                justify="center",
                style={"height": "1%"},
                align="end",
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.I(
                                    children=" ",
                                    style={
                                        "color": "black",
                                        "text-align": "center",
                                        "font-size": "20px",
                                        "font-weight": "Normal",
                                    },
                                    className="fas fa-cog",
                                    id="target_lookout_cog",
                                ),
                                html.I(
                                    children="Lookout Plots",
                                    style={
                                        "color": "black",
                                        "text-align": "center",
                                        "font-size": "20px",
                                        "font-weight": "Normal",
                                        "padding-bottom": "20px",
                                    },
                                    className="fas fa-question-circle fa-lg",
                                    id="target_lookout",
                                ),
                                dbc.Tooltip(
                                    explanations.LOOKOUT_WHAT,
                                    target="target_lookout",
                                    placement="top-start",
                                ),
                                dbc.Tooltip(
                                    explanations.LOOKOUT_HOW,
                                    target="target_lookout_cog",
                                    placement="top-start",
                                ),
                                html.Br(),
                                html.Label(["Budget"], style={"font-weight": "bold"}),
                                dcc.Dropdown(
                                    options=[
                                        {"label": i, "value": i} for i in range(1, 6)
                                    ],
                                    value="2",
                                    id="budget_dropdown",
                                    style={
                                        "width": "50%",
                                        "offset": 1,
                                    },
                                    clearable=False,
                                ),
                            ]
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    "Inliers\n",
                                    style={"color": "rgb(105,105,105)", "font-weight": "bold"},
                                ),
                                html.Br(),
                                html.Span(
                                    "Outliers not maxplained\n",
                                    style={
                                        "color": "rgb(60, 179, 113)",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.Br(),
                                html.Span(
                                    "Outliers maxplained",
                                    style={
                                        "color": "rgb(220, 20, 60)",
                                        "font-weight": "bold",
                                    },
                                ),
                            ]
                        ),
                        width=2,
                    ),
                ],
                style={
                    "background-color": "white",
                    "border-radius": "15px",
                    "padding": "5px",
                },
            ),
            dbc.Row(
                [
                    html.Div(id="lookout", children=[]),
                ],
                justify="center",
                style={
                    "background-color": "white",
                    "border-radius": "15px",
                    "padding": "5px",
                },
            ),
            modal,
        ],
        style={"padding": 10, "background-color": "#EEEEEE"},
    )
    output = html.Div([navbar, body])

    return output
