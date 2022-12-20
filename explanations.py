CLUSTER_EXPLAIN_WHAT = """
What:
Cluster graph shows the clustering of anomaly points. 
The optimal number of clusters is set on default, 
but users can change the number of clusters using the dropdown. 
The color of the clusters distinguishes the data_cancer points by their anomaly score.
As per the color bar on the right, darker the color, higher is the anomaly score.
The shape of the clusters distinguishes the data_cancer points by their cluster group.
X and Y axes are the features obtained after performing Multidimensional Scaling on the original features.
"""

CLUSTER_EXPLAIN_HOW = """
How:
Click on the dropdown to select number of clusters to display.
Click on any data_cancer point to check the top 10 important features
that explain why that point is an anomaly.
"""

HISTOGRAM_EXPLAIN_WHAT = """
what:
Histogram plot displays the frequency distribution of any feature. 
The frequency distribution of the normal points
(depicted in blue) and the anomaly points (depicted in red) 
within the selected cluster for that feature are displayed.
"""
HISTOGRAM_EXPLAIN_HOW = """
How:
Select any feature from the dropdown to get its frequency distribution.
"""

DENSITY_EXPLAIN_WHAT = """
What:
Density plot can be used to explore two features simultaneously. 
The plot displays the density distribution of the data_cancer for two features.
"""

DENSITY_EXPLAIN_HOW = """
How:
Click on the two dropdowns on the right to select two features.
"""

PARALLEL_EXPLAIN_WHAT = """
What:
Parallel Coordinates plot can be used to explore multiple features simultaneously. 
The graph displays the feature values for both the normal points (depicted in blue) 
and anomaly points (depicted in red) within the selected cluster.
"""

PARALLEL_EXPLAIN_HOW = """
How:
The number of features to explore and the specific features can 
be added from the dropdown on the right.
"""

RULE_MINING_WHAT = """
What:
Coverage score:what fraction of cluster-outliers obey or pass the rule
Purity score:what fraction of all the covered points are anomaly points 
For the selected coverage and purity scores, the top rule candidates are displayed.
"""

RULE_MINING_HOW = """
How:
Type in the requires coverage and purity scores, then click “Show rules” for the rules to display.
"""

SCORE_CALCULATION_WHAT = """
What:
User can create rules by adding features and their ranges. Accordingly, 
the coverage and purity scores for the user selected rules is displayed.
"""

SCORE_CALCULATION_HOW = """
How:
Select a particular rule from the left, or create new rules. 
To create a new rule, select the required feature from the dropdown and click on “Add”. 
To change feature values, drag the slider to required values. 
Click on “Calculate Scores” to display coverage and purity scores for the custom rules.
"""

LOOKOUT_WHAT = """
What:
The LookOut algorithm shows the pair of features plot that 'maxplain' the anomalies detected
by the model or the graph that depicts better the outliers. The LookOut model is called the budget,
which is the number of charts displayed that 'maxplain' better the anomalies. 
The black dots are all the inliers, the blue points are the outliers detected by the model
 but not 'maxplained' by the graph, and the red points are the outliers that are 'maxplained' by the graph. 
"""

LOOKOUT_HOW = """
How:
Click on the dropdown menu to select the budget you want to display or the
graphs you want to show that depict better the anomalies detected by the model.
"""
