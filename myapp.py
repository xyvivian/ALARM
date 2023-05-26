import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder
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
from copy import deepcopy

from utils import preprocess, utils
import sys
sys.path.append("xpacs_offline")
from xpacs import *



class Parameters():
    def __init__(self, 
                 json_file_name,
                 ):
        with open(json_file_name, 'r') as openfile:
            json_ = json.load(openfile)
        self.input_file = json_["input_file"][3:]
        self.dataset_name = json_["dataset_name"]
        self.output_file = json_["output_file"][3:]
        self.has_label = json_["has_label"]

parameters = Parameters("parameters.json")
#feature_names = None
dataset_name = parameters.dataset_name
output_file = parameters.output_file

#load the data, X: features, y:label, data index and feature names
X, data_index ,feature_names = utils.get_driver(parameters.input_file, parameters.has_label)
all_result = utils.load_all_result("data/%s/concatenate_result.txt" % dataset_name)

#Set GLOBAL variable - FEAT_NAME (feature names)
FEAT_NAME = feature_names
#print(FEAT_NAME)

all_feature_lst = list(X.columns)
cat_dim_lst = []
for idx,ival in enumerate(X.iloc[0]):
    if type(ival) == str:
        cat_dim_lst.append(list(X.columns)[idx])

#Load explanation values
explanation_value = utils.get_inference(all_result= all_result, 
                                        X= X,
                                        feature_names=feature_names)

#Load scores
score_data = utils.get_anomaly_score_data(all_result = all_result)

normal_index, anomaly_index,anomaly_col = utils.normal_anomaly_idx_split(score_data, data_index)

anomaly_col = pd.DataFrame(anomaly_col,columns = ["anomaly"])
anomaly_col.index = X.index

normal_X = X.iloc[normal_index]
#print("normal")
#print(normal_X.head())
anomaly_X = X.iloc[anomaly_index]
#print(top_k_features)
#print(anomaly_index)



DASHBOARD_DATA = preprocess.prepare_data(
                  all_result = all_result,
                  original_data = X,
                  feature_names=feature_names,
                  cat_dim_lst = cat_dim_lst,
                  anomaly_col = anomaly_col,
                  explanation_value = explanation_value)
#print("Overall dtypes")
#print((DASHBOARD_DATA.graph_data.dtypes))
#print(DASHBOARD_DATA.parallel_data.dtypes)
#print(DASHBOARD_DATA.exploration_data.dtypes)
#print(DASHBOARD_DATA.dropdown_options)



ASSETS_PATH =  dataset_name + "/"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], suppress_callback_exceptions=True)
app.layout = layout(DASHBOARD_DATA)


# add callback for toggling the collapse on small screens
# this is for toggling the small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


"""
Call back for scatter plot
Input: dropdown-specify the number of clusters
Output: the scatter plot on the top left
"""
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

"""
Call back for the histogram plot
# Input: selected_K: dropdown-specify the number of clusters
         cluster_number: dropdown-specify the cluster_idx in the clusters
         histogram_selected_feature: dropdown-specify the features to show in density
  Output: histogram
"""
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
        DASHBOARD_DATA.exploration_data["anomaly"] == 0
    ]

    histogram_data = pd.concat( [histogram_data,
        DASHBOARD_DATA.exploration_data.iloc[selected_indices][
            (DASHBOARD_DATA.exploration_data["anomaly"] == 1)]], axis= 0) 
    
    fig = px.histogram(
        histogram_data,
        x=histogram_selected_feature,
        color="anomaly",
        barmode="overlay",
        color_discrete_sequence=["rgb(0,255,255)", "rgb(255,0,0)"],
    )

    return fig


"""
Call back for update the dropdown option
"""
@app.callback(
    Output(component_id="cluster_dropdown", component_property="options"),
    Input(component_id="dropdown", component_property="value"),
)
def update_date_dropdown(cluster_val):
    cluster_num = int(cluster_val[0])-1
    return [{'label': i, 'value': i} for i in DASHBOARD_DATA.cluster_options[cluster_num]]


"""
Call back for the density plot
# Input: selected_K: dropdown-specify the number of clusters
         cluster_number: dropdown-specify the cluster_idx in the clusters
         density_feature_x: dropdown-specify the X axis of the density plot
         density_feature_y: dropdown-specify the Y axis of the density plot
  Output: density
"""
@app.callback(
    Output(component_id="density", component_property="figure"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input(component_id="dropdown_density1", component_property="value"),
    Input(component_id="dropdown_density2", component_property="value"),
)

def update_density(selected_k, cluster_number, density_feature_x, density_feature_y):
    # selected anomaly points
    selected_indices = DASHBOARD_DATA.graph_data[
        DASHBOARD_DATA.graph_data[selected_k] == int(cluster_number[-1])
    ].index
    # normal data
    density_data = DASHBOARD_DATA.exploration_data
    # anomaly points
    density_data2 = DASHBOARD_DATA.exploration_data.iloc[selected_indices][
            (DASHBOARD_DATA.exploration_data["anomaly"] == 1)]
    
    #show normal points
    fig1 = px.density_contour(density_data, x=density_feature_x, y=density_feature_y)
    fig1.update_traces(
        contours_coloring="fill", colorscale="Portland", contours_showlabels=True
    )
    #show anomaly points
    fig2 = px.scatter(density_data2,
        x=density_feature_x,
        y=density_feature_y,
        color_discrete_sequence=["red"],
    )
    fig = go.Figure(data=fig1.data + fig2.data)
    fig.update_layout(xaxis_title=density_feature_x, yaxis_title=density_feature_y)

    return fig


"""
Call back for the parallel plot
  Input: selected_K: dropdown-specify the number of clusters
         cluster_number: dropdown-specify the cluster_idx in the clusters
         delected_feautres: a list of featurs to show in parallel plot 
  Output: parallel
"""
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
    
    # the same as the exploration data, but all categorical features
    # are encoded into integer values using sklearn.LabelEncoder
    parallel_data = DASHBOARD_DATA.parallel_data[
        DASHBOARD_DATA.parallel_data["anomaly"] == 0
    ]
    
    parallel_data = pd.concat([parallel_data,
        DASHBOARD_DATA.parallel_data.iloc[selected_indices][
            (DASHBOARD_DATA.parallel_data["anomaly"] == 1)]], axis = 0)

    # show the parallel figure
    fig = px.parallel_coordinates(
        parallel_data,
        color="anomaly",
        dimensions=selected_features,
        color_continuous_scale=[(0.00, "rgb(0,255,255)"), (1.00, "rgb(255,0,0)")],
    )
    
    
    # the strained sklearn.LabelEncoder
    encoders = DASHBOARD_DATA.encoders
    dims = []
    # change each categorical feature ticks's from integer values to original categorical features(str)
    for plt_val in fig.to_dict()["data"][0]["dimensions"]:
        if plt_val['label'] in cat_dim_lst:
            map_val = {"tickvals": parallel_data[plt_val['label']].unique(), "ticktext":encoders[plt_val['label']].classes_}
            dims.append({**plt_val, **map_val})
        else:
            map_val = {"tickvals": np.linspace(np.min(parallel_data[plt_val['label']]), np.max(parallel_data[plt_val['label']]),5)}
            dims.append({**plt_val, **map_val})

    fig.update_traces(dimensions=dims)

    return fig

# the dropdown for parallel features
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

"""
Call back to update the number of features and each feature selection in the parallel plots
Input: num_parallel_features: number of features in the aprallel plot
       children: dropdown options before updates
Output: dropdown options after the updates (the features are added using dynamic_control_maker function)
"""
@app.callback(
    Output(component_id="parallel_features", component_property="children"),
    Input(component_id="parallel_feature_num", component_property="value"),
    State("parallel_features", "children"),
)
def update_parallel_feature_num(num_parallel_features, children):
    children = []
    for i in range(1, int(num_parallel_features) + 1):
        children.append(dynamic_control_maker(str(i)))
    return children


"""
Add the pre-processed lookout image to the panel
"""
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


"""
Call back to update the lookout plot
Input: selected_K: dropdown-specify the number of clusters
       cluster_number: dropdown-specify which cluster in the clusters
       budget: dropdown-allow the user to see how many plots
Output: A list of Lookout plots which explain the selected cluster of outliers
"""
@app.callback(
    Output(component_id="lookout", component_property="children"),
    Input(component_id="dropdown", component_property="value"),
    Input(component_id="cluster_dropdown", component_property="value"),
    Input(component_id="budget_dropdown", component_property="value"),
)
def update_lookout(selected_k, cluster_number, budget):
    kmeans_id = selected_k.split(" ")[0]
    cluster_id = str(int(cluster_number[7:]))
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
        if "," in trigger["prop_id"]:         
            idx_str = trigger["prop_id"].split(",")[0]
            if ":"  in idx_str:
                idx = idx_str.split(":")[1].replace('"', "")
                new_children = []
                for child in children:
                    #print(child["props"]["children"][0]["props"]["children"][0].replace(" ",""))
                    if child["props"]["children"][0]["props"]["children"][0].replace(" ","") != idx:
                        new_children.append(child)
                children = new_children
                #print("new children")
                #print(new_children)
    
    else:

        if selected_rule not in cat_dim_lst:
            feature_min = round(
            DASHBOARD_DATA.exploration_data[selected_rule].min(axis=0), 4
            )
            feature_max = round(
            DASHBOARD_DATA.exploration_data[selected_rule].max(axis=0), 4
            )
            children.append(
            rule_slider_maker(
                selected_rule, feature_min, feature_max, feature_min, feature_max)
            )
        else:
            all_vals = list(DASHBOARD_DATA.exploration_data[selected_rule].unique())
            children.append(
            rule_dropdown_maker(selected_rule, all_vals[0], all_vals)
            )
    #print("final children")
    #print(children)
    return children


"""
Calculate scores based on updated rules
Input: n_clicks: 
       rule_sliders: experts' modified rules
       selected_rule: original/default rules
       group_num: which group of the cluster
       cluster_num: number of clusters
Output: coverage and purity calculated of the selected rules + updated rules
"""
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
    group0_anomaly = pd.read_csv('xpacs_offline/results/%s/cluster%d_idx%d_group_anomaly.txt' \
                                 %(dataset_name, kmeans_id, cluster_id),index_col = 0)
    group0_anomaly = group0_anomaly.reset_index()
    del group0_anomaly['index']
    normal_X = pd.read_csv('xpacs_offline/results/%s/normal_data.txt' % dataset_name)
    combined_rules = []
    
    for child in selected_rules:
        feat = (child["props"]["children"][0]["props"]["children"][0].replace(" ", ""))
        id = FEAT_NAME.index(feat)
        interval = child["props"]["children"][1]["props"]["value"]
        if type(interval) == list:
            combined_rules.append((id, (interval[0], interval[1])))
        else:
            combined_rules.append((id, (interval, interval)))
    
    #print("combined_rules")
    #print(combined_rules)     
   
    #print("rule_slider")
    for child in rules_slider:
        #print(child)
        feat = (child["props"]["children"][0]["props"]["children"][0].replace(" ", ""))
        id = FEAT_NAME.index(feat)
        interval = child["props"]["children"][1]["props"]["value"]
        if type(interval) == list:
            combined_rules.append((id, (interval[0], interval[1])))
        else:
            combined_rules.append((id, (interval, interval)))
    #print("total rules!")
    #print(combined_rules)
    combined_rules = remove_redundant_candidates(combined_rules)
    (mass, purity) = get_mass_purity(group0_anomaly, normal_X, combined_rules,FEAT_NAME,cat_dim_lst)
    return html.Div("Current COVERAGE: %.3f, Current PURITY:%.3f " % (mass, purity))



"""
Update the rule candidates based on the expert's typing into the Coverage and Purity threshold
on the middle panel of the main panel.
Input: submit-rules click
       purity: a purity value input by the user
       mass: a mass value input by the user
       group_num: which cluster number 
       cluster_num: how many clusters
Output: display of top 3 rules that satisfy the user's selection -> should be html.div
"""
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
            "xpacs_offline/results/%s/cluster%d_idx%d_candidates.txt"
            % (dataset_name,kmeans_id, cluster_id),
            "rb",
        ) as f:
            candidates = pickle.load(f)

        group0_anomaly = pd.read_csv('xpacs_offline/results/%s/cluster%d_idx%d_group_anomaly.txt'%(dataset_name, kmeans_id, cluster_id),index_col = 0)
        group0_anomaly = group0_anomaly.reset_index()
        del group0_anomaly['index']
        normal_X = pd.read_csv('xpacs_offline/results/%s/normal_data.txt' % dataset_name)
        feature_names = list(normal_X.columns)
        found_results = find_displaying_candidate_above_threshold(group0_anomaly,
                                        normal_X, 
                                        candidates=candidates[0],
                                        candidate_scores = candidates[1], 
                                        mass=mass, 
                                        purity=purity)
        
        results = found_results[0]
        scores = found_results[1]
        children = []
        for i in range(len(results)):
            result = results[i]
            score = scores[i]

            children.append(dynamic_card_maker(result,score, i,feature_names))
        if len(children) == 0:
            return html.Div("No candidates, try modifying coverage and purity percentage.")
    else:
        return html.Div("No purity and coverage provided")
    return children


"""
Create a dynamic card with rules, mass and purity
Input: result: rules
       score: mass, purity
       i: result index
       feature_names: list of feature names to display
"""
def dynamic_card_maker(result, score, i,feature_names):
    body = []
    for ele in result:
        feat_name = feature_names[int(ele[0])]
        if feat_name not in cat_dim_lst:
            body.append(
            html.Div(
                "Feature "+
                feat_name.upper()
                + " is between: "
                + "{:.3f}".format(ele[1][0])
                + " and "
                + "{:.3f}".format(ele[1][1])
            ))
        else:
            body.append(
            html.Div(
                "Feature "+
                feat_name.upper()
                + " is: "
                + "%s" % ele[1][0]
        ))
    # append the score and purity selection        
    body.append(
     html.Div(
         "Score: "+
         "Coverage is "+ 
         "{:.3f}".format(score[0])+
          " and Purity is "+
         "{:.3f}".format(score[1])
         ))
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
    
    button_type = ""
    button_str_lst = trigger["prop_id"].split(",")
    if len(button_str_lst) > 1:
        button_str_lst = button_str_lst[1].split(":")
        if len(button_str_lst) > 1:
            button_type = button_str_lst[1]
    
    
    if "dynamic-select-button" in button_type:
        index = trigger["prop_id"].split(",")[0].split(":")[1].replace('"', "")
        child = children1[int(index)]
        #print(child)
        children1 = []
        rule_feat_names = []
        for ele in child["props"]["children"]["props"]["children"][:-1]:
            feature = ele["props"]["children"]
            if not feature.startswith("Feature "):
                continue
            feature_name = feature[feature.index("Feature ") + len("Feature ") : feature.index("is") - 1].lower()
            feat_name = feature_name.replace(" ", "").lower()
            rule_feat_names.append(feat_name)

            #print(feat_name)
            if feat_name not in cat_dim_lst:
                feature_start = round(
                float(feature[feature.index("between") + len("between")+1 : feature.index("and") - 1]),
                3,
                )
                feature_end = round(
                float(feature[feature.index("and") + 4 : len(feature)]), 3
                )
                feature_max = round(
                    DASHBOARD_DATA.exploration_data[feature_name.replace(" ", "")].max(
                    axis=0
                ),
                3,
                )
                feature_min = round(
                    DASHBOARD_DATA.exploration_data[feature_name.replace(" ", "")].min(
                    axis=0
                ),
                3,
                )
                children1.append(
                    rule_slider_maker(
                    feature_name, feature_min, feature_max, feature_start, feature_end
                )
                )
            else:
                feature_val = str(feature[feature.index("is: ") + len("is: ") :])
                all_vals = list(DASHBOARD_DATA.exploration_data[feat_name].unique())
                children1.append(
                    rule_dropdown_maker(feature_name, feature_val, all_vals)
                    )
        return children1
    else:
        idx_str_lst = trigger["prop_id"].split(",")[0].split(":")
        #value_ = trigger["value"]
        idx =""
        if len(idx_str_lst)> 1:
            idx = idx_str_lst[1].replace('"', "")
        new_children = []
        children_indices = []
        for child in children2:
            child_idx = child["props"]["children"][0]["props"]["children"][1]["props"]["id"]["index"]
            n_clicks = child["props"]["children"][0]["props"]["children"][1]["props"]["n_clicks"]
            if child_idx != idx:
                if n_clicks == 0:
                    new_children.append(child)
                    children_indices.append(idx)
        children2 = new_children
        return children2


def rule_slider_maker(name, min_val, max_val, start, end):
    return dbc.Row(
        [
            dbc.Col(
                [
                    name + " ",
                    html.Button(
                        "Delete",
                        id={"type": "dynamic-button", "index": name.replace(" ", "")},
                        n_clicks=0,
                    ),
                ],
                width=3,
            ),
            dcc.RangeSlider(
                min_val,
                max_val,
                value=[start, end],
                marks={
                    min_val: "{:.3f}".format(min_val),
                    start: str(start),
                    end: str(end),
                    max_val: "{:.3f}".format(max_val),
                },
                id={"type": "dynamic-slider", "index": name.replace(" ", "")},
            ),
        ]
    )


def rule_dropdown_maker(name, selected_val, all_vals):
    return dbc.Row(
        [
            dbc.Col(
                [
                    name +" ",
                    html.Button(
                        "Delete",
                        id={"type": "dynamic-button", "index": name.replace(" ", "")},
                        n_clicks=0,
                    ),
                ],
                width=3,
            ),
            dcc.Dropdown(
                options=[
                    {"label": i, "value": i}
                    for i in all_vals
                ],
                value= selected_val,
                id={"type": "dynamic-dropdown", "index": name.replace(" ","")},
                style={
                    "width": "50%",
                    "offset": 1,
                },
                clearable=False,
            ),
        ]
    )


@app.callback(
    Output("output-container-range-slider", "children"),
    Input({"type": "dynamic-slider","index": dash.ALL}, "id"), 
    Input({"type": "dynamic-slider", "index": dash.ALL}, "value"),
    Input({"type": "dynamic-dropdown","index": dash.ALL}, "id"), 
    Input({"type": "dynamic-dropdown","index": dash.ALL}, "value"),
  
)
def update_output(feat1,value1,feat2, value2,):
    updated_str = ""
    
    #print real-valued features and values
    for i in range(len(value1)):
        rule_str = "Feature "
        feature  =  feat1[i]["index"].upper()
        #print(feature)
        new_rul = " between: %.3f and %.3f." % (value1[i][0], value1[i][1])
        rule_str = rule_str + feature + new_rul
        updated_str = updated_str + rule_str
      
    #print categorical features and values
    for i in range(len(value2)):
        rule_str = "Feature "
        feature = feat2[i]["index"].upper()
        #print(feature)
        new_rul = " is: %s." % value2[i]
        rule_str = rule_str + feature + new_rul
        updated_str = updated_str + rule_str
        
    return 'You have selected: "' + updated_str + '"'

    # callbacks


@app.callback(
    [Output("modal", "is_open"), Output("modal_graph", "figure")],
    [Input("scatter", "clickData"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(hover_data, close_button, is_open):
    if hover_data or close_button:
        s = DASHBOARD_DATA.graph_data[
            round(DASHBOARD_DATA.graph_data["x"], 5)
            == round(hover_data["points"][0]["x"], 5)
        ].index[0]
        index = int(s)
        #print(DASHBOARD_DATA.explanation)
        explanations = DASHBOARD_DATA.explanation.loc[int(s)]
        #s = s[DASHBOARD_DATA.features]
        s = explanations.sort_values(ascending=False)
        #s = s[:10]
        #print(s)
        s = pd.DataFrame({"feature": s.index, "importance": s.values})
        fig = px.bar(s, x="importance", y="feature", orientation="h", title= "Feature explanation for index %d" % index)
        return not is_open, fig
    else:
        return is_open, None


"""
Save results to output txt files. -> txt files at default is in the result/rules 
directory, can specify the directroy within parameters.json file
"""
@app.callback(
    Output(component_id="saved", component_property="children"),
    [Input(component_id="save_rules", component_property="n_clicks")],
    [
        State(component_id="rules_slider", component_property="children"),
        State(component_id="selected_rule", component_property="children"),
        State(component_id="dropdown", component_property="value"),
        State(component_id="cluster_dropdown", component_property="value"),
    ],
)
def save_results(n_clicks, rules_slider, selected_rules, group_num, cluster_num):
    if n_clicks < 1:
        return html.Div("")
    kmeans_id = int(group_num.split(" ")[0])
    cluster_id = int(cluster_num[7:])
    #cluster information
    cluster_info = "For %d clusters, cluster index %d is selected. \n" % (kmeans_id, cluster_id)
    
    group0_anomaly = pd.read_csv('xpacs_offline/results/%s/cluster%d_idx%d_group_anomaly.txt' \
                                 %(dataset_name, kmeans_id, cluster_id),index_col = 0)
    group0_anomaly = group0_anomaly.reset_index()
    del group0_anomaly['index']
    normal_X = pd.read_csv('xpacs_offline/results/%s/normal_data.txt' % dataset_name)
    combined_rules = []
    
    for child in selected_rules:
        feat = (child["props"]["children"][0]["props"]["children"][0].replace(" ", ""))
        id = FEAT_NAME.index(feat)
        interval = child["props"]["children"][1]["props"]["value"]
        if type(interval) == list:
            combined_rules.append((id, (interval[0], interval[1])))
        else:
            combined_rules.append((id, (interval, interval)))
    
    for child in rules_slider:
        #print(child)
        feat = (child["props"]["children"][0]["props"]["children"][0].replace(" ", ""))
        id = FEAT_NAME.index(feat)
        interval = child["props"]["children"][1]["props"]["value"]
        if type(interval) == list:
            combined_rules.append((id, (interval[0], interval[1])))
        else:
            combined_rules.append((id, (interval, interval)))

    combined_rules = remove_redundant_candidates(combined_rules)
    #print rule information
    rules_str = ""
    for i in combined_rules:
        rule_str = "Feature "
        rule_id = i[0]
        rule_interval = (i[1][0],i[1][1])
        if FEAT_NAME[rule_id] in cat_dim_lst:
            new_rul = " is: %s." % rule_interval[0]
            rule_str = rule_str + FEAT_NAME[rule_id] + new_rul
        else:
            new_rul = " between: %.3f and %.3f." % (rule_interval[0], rule_interval[1])
            rule_str = rule_str + FEAT_NAME[rule_id] + new_rul
        rules_str = rules_str + rule_str
    rules_str = rules_str + "\n"
    
    (mass, purity) = get_mass_purity(group0_anomaly, normal_X, combined_rules,FEAT_NAME,cat_dim_lst)
    mass_purity_info ="Current COVERAGE: %.3f, Current PURITY:%.3f. \n" % (mass, purity)
    
    with open(output_file, "a+") as myfile:
        myfile.write(cluster_info + rules_str + mass_purity_info +"\n")
    
    return html.Div("Rules saved")




if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
