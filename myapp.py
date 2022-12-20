import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity

import plotly.express as px
import pickle

import itertools
import dash
import dash_labs as dl
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import os
from dash import callback_context

from copy import deepcopy
from myapp import *
from layout import layout

from utils import preprocess, utils

# feature_names = ['feature0','radius_mean', 'texture_mean', 'perimeter_mean',
#                  'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#                  'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#                  'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#                  'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
#                  'fractal_dimension_se', 'radius_worst', 'texture_worst',
#                  'perimeter_worst', 'area_worst', 'smoothness_worst',
#                  'compactness_worst', 'concavity_worst', 'concave_points_worst',
#                  'symmetry_worst', 'fractal_dimension_worst']
#
feature_names = None

data, X, y, data_index = utils.get_driver("data/nullfox/drive_data_with_labels.txt")
outlier_data, shap_inference = utils.get_inference(
    "data/nullfox/drive_feature_inference_results.projection.50.average.top100.result.txt",
    feature_names=feature_names,
)
score_data = utils.get_anomaly_score_data("data/nullfox/drive_anomaly_scores.txt")

normal_index, anomaly_index = utils.normal_anomaly_idx_split(shap_inference, data_index)

normal_X = X[normal_index]
top_k_features = X[anomaly_index]

data_transformed, data_MDS = utils.get_mds(shap_inference)

DASHBOARD_DATA = preprocess.prepare_data(
    data_transformed,
    data_MDS,
    outlier_data,
    score_data,
    data,
    feature_names=feature_names,
    max_k=10,
)
ASSETS_PATH = "nullfox/"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])
app.layout = layout(DASHBOARD_DATA)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output(component_id="scatter", component_property="figure"),
    Input(component_id="dropdown", component_property="value"),
)
def update_scatter(scatter_selected_feature):
    fig = px.scatter(
        DASHBOARD_DATA.graph_data,
        x="x",
        y="y",
        color="score",
        symbol=scatter_selected_feature,
        color_continuous_scale="Portland",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(
    Output(component_id="histogram", component_property="figure"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input(component_id="dropdown_histogram", component_property="value"),
)
def update_histogram(selected_k, cluster_number, histogram_selected_feature):
    selected_indices = DASHBOARD_DATA.graph_data[
        DASHBOARD_DATA.graph_data[selected_k] == int(cluster_number[-1])
    ].index

    histogram_data = DASHBOARD_DATA.exploration_data[
        DASHBOARD_DATA.exploration_data["anamoly"] == 0
    ]

    histogram_data = histogram_data.append(
        DASHBOARD_DATA.exploration_data.iloc[selected_indices][
            (DASHBOARD_DATA.exploration_data["anamoly"] == 1)
        ]
    )
    fig = px.histogram(
        histogram_data,
        x=histogram_selected_feature,
        color="anamoly",
        barmode="overlay",
        color_discrete_sequence=["rgb(0,255,255)", "rgb(255,0,0)"],
    )

    return fig


@app.callback(
    Output(component_id="density", component_property="figure"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input(component_id="dropdown_density1", component_property="value"),
    Input(component_id="dropdown_density2", component_property="value"),
)
def update_density(selected_k, cluster_number, density_feature_x, density_feature_y):
    selected_indices = DASHBOARD_DATA.graph_data[
        DASHBOARD_DATA.graph_data[selected_k] == int(cluster_number[-1])
    ].index
    density_data = DASHBOARD_DATA.exploration_data
    density_data = density_data.append(
        DASHBOARD_DATA.exploration_data.iloc[selected_indices][
            (DASHBOARD_DATA.exploration_data["anamoly"] == 1)
        ]
    )
    fig1 = px.density_contour(density_data, x=density_feature_x, y=density_feature_y)
    fig1.update_traces(
        contours_coloring="fill", colorscale="Portland", contours_showlabels=True
    )
    fig2 = px.scatter(
        density_data[density_data["anamoly"] == 1],
        x=density_feature_x,
        y=density_feature_y,
        color_discrete_sequence=["red"],
    )
    fig = go.Figure(data=fig1.data + fig2.data)
    fig.update_layout(xaxis_title=density_feature_x, yaxis_title=density_feature_y)

    return fig


@app.callback(
    Output(component_id="parallel", component_property="figure"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input({"type": "filter-dropdown", "index": dash.ALL}, "value"),
)
def update_parallel(selected_k, cluster_number, selected_features):
    selected_indices = DASHBOARD_DATA.graph_data[
        DASHBOARD_DATA.graph_data[selected_k] == int(cluster_number[-1])
    ].index
    parallel_data = DASHBOARD_DATA.exploration_data[
        DASHBOARD_DATA.exploration_data["anamoly"] == 0
    ]
    parallel_data = parallel_data.append(
        DASHBOARD_DATA.exploration_data.iloc[selected_indices][
            (DASHBOARD_DATA.exploration_data["anamoly"] == 1)
        ]
    )
    fig = px.parallel_coordinates(
        parallel_data,
        color="anamoly",
        dimensions=[value for value in selected_features],
        color_continuous_scale=[(0.00, "rgb(0,255,255)"), (1.00, "rgb(255,0,0)")],
    )

    return fig


def dynamic_control_maker(val):
    return dcc.Dropdown(
        options=[{"label": i, "value": i} for i in DASHBOARD_DATA.dropdown_options],
        value=DASHBOARD_DATA.dropdown_options[int(val)],
        id={"type": "filter-dropdown", "index": "dropdown_parallel" + val},
        style={
            "offset": 1,
        },
        clearable=False,
    )


@app.callback(
    Output(component_id="parallel_features", component_property="children"),
    Input(component_id="parallel_feature_num", component_property="value"),
    State("parallel_features", "children"),
)
def update_parallel_feature_num(num_parallel_features, children):
    size = len(children)
    for i in range(size + 1, int(num_parallel_features) + 1):
        children.append(dynamic_control_maker(str(i)))
    return children


def dropdown_fig_maker(path):
    return html.Div(
        [
            html.Img(
                src=app.get_asset_url(path),
                alt="lookout for " + path,
                style={"height": "300px", "width": "300px"},
            )
        ],
        style={"display": "inline-block", "margin": "10px"},
    )


@app.callback(
    Output(component_id="lookout", component_property="children"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input(component_id="budget_dropdown", component_property="value"),
)
def update_lookout(selected_k, cluster_number, budget):
    kmeans_id = selected_k.split(" ")[0]
    cluster_id = str(int(cluster_number[7:]) + 1)
    budget = budget
    i = 1
    children = []
    while True:
        path = (
            ASSETS_PATH
            + "lookout-"
            + str(kmeans_id)
            + "-"
            + str(cluster_id)
            + "-"
            + str(budget)
            + "-"
            + str(i)
            + ".png"
        )
        if os.path.isfile("assets/" + path):
            children.append(dropdown_fig_maker(path))
            i += 1
        else:
            break
    return children


@app.callback(
    Output(component_id="rules_slider", component_property="children"),
    Input(component_id="submit-val", component_property="n_clicks"),
    Input({"type": "dynamic-button", "index": dash.ALL}, "n_clicks"),
    Input(component_id="feature_dropdown", component_property="value"),
    State("rules_slider", "children"),
)
def update_rules(feature1, feature2, selected_rule, children):
    trigger = callback_context.triggered[0]
    if trigger["prop_id"] != "submit-val.n_clicks":
        idx = trigger["prop_id"].split(",")[0].split(":")[1].replace('"', "")
        new_children = []
        for child in children:
            if child["props"]["children"][0]["props"]["children"][0] != idx:
                new_children.append(child)
        children = new_children
    else:
        feature_min = round(
            DASHBOARD_DATA.exploration_data[selected_rule].min(axis=0), 4
        )
        feature_max = round(
            DASHBOARD_DATA.exploration_data[selected_rule].max(axis=0), 4
        )
        children.append(
            rule_slider_maker(
                selected_rule, feature_min, feature_max, feature_min, feature_max
            )
        )
    return children


@app.callback(
    Output(component_id="scores", component_property="children"),
    [Input(component_id="score_button", component_property="n_clicks")],
    [
        State(component_id="rules_slider", component_property="children"),
        State(component_id="selected_rule", component_property="children"),
        State(component_id="dropdown", component_property="value"),
        State(component_id="cluster_dropdown", component_property="value"),
    ],
)
def calculate_scores(n_clicks, rules_slider, selected_rules, group_num, cluster_num):
    kmeans_id = int(group_num.split(" ")[0])
    cluster_id = int(cluster_num[7:])
    group0_anomaly = np.load(
        "xpacs_offline/results/cluster%d_idx%d_group_anomaly.npy"
        % (kmeans_id, cluster_id)
    )
    normal_X = np.load("xpacs_offline/results/normal.npy")

    combined_rules = []
    for child in selected_rules:
        id = int(child["props"]["children"][0]["props"]["children"][0].split(" ")[1])
        interval = child["props"]["children"][1]["props"]["value"]
        combined_rules.append((id - 1, (interval[0], interval[1])))
    for child in rules_slider:
        id = int(child["props"]["children"][0]["props"]["children"][0][-1])
        interval = child["props"]["children"][1]["props"]["value"]
        combined_rules.append((id - 1, (interval[0], interval[1])))
    combined_rules = remove_redundant_candidates(combined_rules)
    (mass, purity) = get_mass_purity(group0_anomaly, normal_X, combined_rules)
    return html.Div("Current COVERAGE: %.4f, Current PURITY:%.4f " % (mass, purity))


# callbacks
@app.callback(
    Output(component_id="rules", component_property="children"),
    [Input(component_id="submit-rules", component_property="n_clicks")],
    [
        State(component_id="purity_input", component_property="value"),
        State(component_id="mass_input", component_property="value"),
        State(component_id="dropdown", component_property="value"),
        State(component_id="cluster_dropdown", component_property="value"),
    ],
)
def update_rule_candidates(n_clicks, purity, mass, group_num, cluster_num):
    if mass and purity:
        kmeans_id = int(group_num.split(" ")[0])
        cluster_id = int(cluster_num[7:])
        with open(
            "xpacs_offline/results/cluster%d_idx%d_candidates.txt"
            % (kmeans_id, cluster_id),
            "rb",
        ) as f:
            candidates = pickle.load(f)

        group0_anomaly = np.load(
            "xpacs_offline/results/cluster%d_idx%d_group_anomaly.npy"
            % (kmeans_id, cluster_id)
        )
        normal_X = np.load("xpacs_offline/results/normal.npy")

        found_results = find_top_k_candidate_above_threshold(
            group0_anomaly,
            normal_X,
            candidates[0],
            candidates[1],
            k=3,
            mass=mass,
            purity=purity,
        )
        results = found_results[0]
        children = []
        for i in range(len(results)):
            result = results[i]
            children.append(dynamic_card_maker(result, i))
        if len(children) == 0:
            return html.Div("No candidates, try modifying mass and purity percentage.")
    else:
        return html.Div("No purity and coverage provided")
    return children


def dynamic_card_maker(result, i):
    body = []
    for ele in result:
        body.append(
            html.Div(
                "feature "
                + str(ele[0] + 1)
                + " is between "
                + "{:.4f}".format(ele[1][0])
                + " and "
                + "{:.4f}".format(ele[1][1])
            )
        )
    body.append(
        html.Button(
            "Select", id={"type": "dynamic-select-button", "index": i}, n_clicks=0
        )
    )
    return dbc.Card(dbc.CardBody(body), id="card" + str(i))


@app.callback(
    Output(component_id="selected_rule", component_property="children"),
    Input({"type": "dynamic-select-button", "index": dash.ALL}, "n_clicks"),
    Input({"type": "dynamic-button", "index": dash.ALL}, "n_clicks"),
    [
        State(component_id="rules", component_property="children"),
        State(component_id="selected_rule", component_property="children"),
    ],
)
def update_rule_candidates1(feature1, feature2, children1, children2):
    trigger = callback_context.triggered[0]
    button_type = trigger["prop_id"].split(",")[1].split(":")[1]
    if "dynamic-select-button" in button_type:
        index = trigger["prop_id"].split(",")[0].split(":")[1].replace('"', "")
        child = children1[int(index)]
        children1 = []
        for ele in child["props"]["children"]["props"]["children"][:-1]:
            feature = ele["props"]["children"]
            feature_name = feature[0 : feature.index("is") - 1]
            feature_start = round(
                float(feature[feature.index("between") + 8 : feature.index("and") - 1]),
                4,
            )
            feature_end = round(
                float(feature[feature.index("and") + 4 : len(feature)]), 4
            )
            feature_max = round(
                DASHBOARD_DATA.exploration_data[feature_name.replace(" ", "")].max(
                    axis=0
                ),
                4,
            )
            feature_min = round(
                DASHBOARD_DATA.exploration_data[feature_name.replace(" ", "")].min(
                    axis=0
                ),
                4,
            )
            children1.append(
                rule_slider_maker(
                    feature_name, feature_min, feature_max, feature_start, feature_end
                )
            )
        return children1
    else:
        idx = trigger["prop_id"].split(",")[0].split(":")[1].replace('"', "")
        new_children = []
        for child in children2:
            if (
                child["props"]["children"][0]["props"]["children"][1]["props"]["id"][
                    "index"
                ]
                != idx
            ):
                new_children.append(child)
        children2 = new_children
        return children2


def rule_slider_maker(name, min_val, max_val, start, end):
    return dbc.Row(
        [
            dbc.Col(
                [
                    name,
                    html.Button(
                        "Delete",
                        id={"type": "dynamic-button", "index": name.replace(" ", "")},
                        n_clicks=0,
                    ),
                ],
                width=2,
            ),
            dcc.RangeSlider(
                min_val,
                max_val,
                value=[start, end],
                marks={
                    min_val: "{:.4f}".format(min_val),
                    start: str(start),
                    end: str(end),
                    max_val: "{:.4f}".format(max_val),
                },
                id={"type": "dynamic-slider", "index": name.replace(" ", "")},
            ),
        ]
    )


@app.callback(
    Output("output-container-range-slider", "children"),
    Input({"type": "dynamic-slider", "index": dash.ALL}, "value"),
)
def update_output(value):
    for i in range(len(value)):
        value[i][0] = round(value[i][0], 4)
        value[i][1] = round(value[i][1], 4)
    return 'You have selected "{}"'.format(value)

    # callbacks


@app.callback(
    [Output("modal", "is_open"), Output("modal_graph", "figure")],
    [Input("scatter", "clickData"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(hover_data, close_button, is_open):
    if hover_data or close_button:
        s = DASHBOARD_DATA.graph_data[
            round(DASHBOARD_DATA.graph_data["x"], 3)
            == round(hover_data["points"][0]["x"], 3)
        ].iloc[0]
        s = s[DASHBOARD_DATA.features]
        s = s.sort_values(ascending=False)
        s = s[:10]
        s = pd.DataFrame({"feature": s.index, "importance": s.values})
        fig = px.bar(s, x="importance", y="feature", orientation="h")
        return not is_open, fig


def find_top_k_candidates(num_clusters, cluster_id, pu_percentage, ms_percentage, k=3):
    kmeans_y = DASHBOARD_DATA.label_lst[num_clusters - 2]
    group_anomaly = top_k_features[kmeans_y == cluster_id]
    ms = ms_percentage * group_anomaly.shape[0]
    mu = (1 - pu_percentage) * normal_X.shape[0]
    explanation_anomaly = DASHBOARD_DATA.explanation_value[kmeans_y == cluster_id]
    index_lst = np.argsort(np.mean(explanation_anomaly, axis=0))[::-1].tolist()[:10]

    rules = find_hyper_rectangles(index_lst, group_anomaly)
    R_candidates = deepcopy(rules)
    candidates = find_all_candidates(R_candidates, group_anomaly, normal_X, mu, ms)
    mass_purity_lst = []
    for i in candidates:
        anomaly_indx = anomaly_index_above_threshold(group_anomaly, i)
        normal_indx = anomaly_index_above_threshold(normal_X, i)
        mass = len(anomaly_indx)
        purity = len(normal_indx)
        mass_purity_lst.append([mass, purity])

    indexed_mass_purity_lst = list(
        zip(list(range(len(mass_purity_lst))), mass_purity_lst)
    )
    val = sorted(indexed_mass_purity_lst, key=lambda sl: (-sl[1][0], sl[1][1]))
    found_index = []
    for i in range(k):
        found_index.append(val[i][0])
    return [candidates[i] for i in found_index]


def find_all_candidates(R_candidates, group_anomaly, normal_X, mu, ms):
    R = []
    for repeat in range(5):
        R_pure = []
        R_non_pure = []
        for cur_rul in R_candidates:
            anomaly_indx = anomaly_index_above_threshold(group_anomaly, cur_rul)
            normal_indx = anomaly_index_above_threshold(normal_X, cur_rul)
            mass = len(anomaly_indx)
            purity = len(normal_indx)
            if mass >= ms:
                if purity < mu:
                    R_pure.append(cur_rul)
                else:
                    R_non_pure.append(cur_rul)
        if len(R_pure) != 0:
            R.extend(R_pure)
        if repeat == 0:
            R_val = R_pure + R_non_pure
        R_candidates = generate_candidate(R_val, repeat)
        R_candidates = remove_redundant_candidates(R_candidates)
    return remove_redundant_candidates(R)


def generate_candidate(R_val, repeat):
    R_candidate = R_val
    all_possible_R_lst = []
    for possible_R in itertools.combinations(R_candidate, repeat + 2):
        all_possible_R_lst.append(merge_thresholds(possible_R[0], possible_R[1]))
    return all_possible_R_lst


def remove_redundant_candidates(lst):
    b_set = set(tuple(x) for x in lst)
    b = [list(x) for x in b_set]
    return b


def union_lst(threshold_union_lst):
    union_results = []
    quantile_ = []
    value_ = []
    cur_index = threshold_union_lst[0][0]
    for i in threshold_union_lst:
        quantile_.append("l")
        value_.append(i[1][0])
        quantile_.append("r")
        value_.append(i[1][1])
    sorted_value_, sorted_quantile_ = zip(
        *sorted(zip(value_, quantile_), key=lambda x: x[0])
    )
    for i in range(len(sorted_value_) - 1):
        cur_value = sorted_value_[i]
        next_value = sorted_value_[i + 1]
        if sorted_quantile_[i] == "l" and sorted_quantile_[i + 1] == "r":
            union_results.append((cur_index, (cur_value, next_value)))
    return union_results


def merge_thresholds(threshold_1, threshold_2):
    threshold_lst = threshold_1 + threshold_2
    result = []

    index_set = set()
    for item in threshold_lst:
        index_set.add(item[0])

    for index in index_set:
        threshold_union_lst = [i for i in threshold_lst if i[0] == index]
        new_union_lst = union_lst(threshold_union_lst)
        result.extend(new_union_lst)
    return result


def find_hyper_rectangles(index_lst, group_anomaly):
    rules = []
    for index in index_lst:
        new_X = group_anomaly[:, index : index + 1]
        X_plot = np.linspace(-0.01, np.max(new_X) + 0.005, 1000)[:, np.newaxis]

        color = "darkorange"
        kernel = "gaussian"

        kde = KernelDensity(kernel=kernel, bandwidth=0.005).fit(new_X)
        log_dens = kde.score_samples(X_plot)

        quantile = [80, 85, 90, 95]
        color = ["red", "limegreen", "violet", "deepskyblue"]

        for i, c in zip(quantile, color):
            thresholds = get_quantile_info(X_plot, index, log_dens, i)
            rules.extend(thresholds)

    return rules


def get_quantile_info(X, index, log_dens, percentage):
    def find_all_thresholds(lst):
        quantile_lst = []
        left_ = 0
        right_ = 0
        for i in range(len(lst)):
            if i == 0:
                left_ = lst[i]
            else:
                if lst[i - 1] + 1 != lst[i]:
                    quantile_lst.append((left_, right_))
                    left_ = lst[i]
            right_ = lst[i]
        quantile_lst.append((left_, right_))
        return quantile_lst

    dens = np.exp(log_dens)
    candidate_lst = (dens > (percentage / 100) * np.max(dens)).nonzero()[0].tolist()
    threshold_indices = find_all_thresholds(candidate_lst)
    true_threshold = [[(index, (X[i[0]][0], X[i[1]][0]))] for i in threshold_indices]
    return true_threshold


def anomaly_index_above_threshold(X, threshold):
    anomaly_indices = {}
    for index, thres in threshold:
        if index not in anomaly_indices.keys():
            anomaly_indices[index] = np.where(
                (X[:, index] >= thres[0]) & (X[:, index] <= thres[1])
            )[0].tolist()
        else:
            anomaly_indices[index].extend(
                np.where((X[:, index] >= thres[0]) & (X[:, index] <= thres[1]))[
                    0
                ].tolist()
            )

    ads = list(anomaly_indices.values())
    return list(set(ads[0]).intersection(*ads))


def find_top_k_candidate_above_threshold(
    group_anomaly, normal_X, candidates, candidate_scores, mass=0.3, purity=0.91, k=3
):
    candidate_lst = []
    mass_purity_lst = []
    for index, i in enumerate(candidate_scores):
        mass_i = i[0]
        purity_i = i[1]
        if mass_i >= mass and purity_i >= purity:
            candidate_lst.append(candidates[index])
            mass_purity_lst.append([mass_i, purity_i])
    if len(mass_purity_lst) == 0:
        return [], []
    indexed_mass_purity_lst = list(
        zip(list(range(len(mass_purity_lst))), mass_purity_lst)
    )
    val = sorted(indexed_mass_purity_lst, key=lambda sl: (-sl[1][0], -sl[1][1]))
    found_index = []
    for i in range(k):
        if i >= len(val):
            break
        found_index.append(val[i][0])
    return [candidate_lst[i] for i in found_index], [
        mass_purity_lst[i] for i in found_index
    ]


def get_mass_purity(group_anomaly, normal_X, candidate_rule):
    anomaly_indx = anomaly_index_above_threshold(group_anomaly, candidate_rule)
    normal_indx = anomaly_index_above_threshold(normal_X, candidate_rule)
    mass = len(anomaly_indx)
    purity = len(normal_indx)
    return (mass / group_anomaly.shape[0], 1 - purity / normal_X.shape[0])


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
