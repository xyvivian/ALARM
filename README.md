# guidance for running

1. cd config
2. make sure your conda version is compatible with python version = 3.9
3. conda env create -f env.yml (creates a environment called iarm)
4. source activate iarm

(if the input data file is not .txt/ .csv with "index" as the indexing_col, please reformat it using
the jupyter notebook data/process_data_add_index.ipynb)

5. cd ..
6. cd xstream
7. change the xstream_parameters.json file with parameters and dataset you have selected
   (refer to xstream/helper.md) for details
8. python xstream_with_explanation.py xstream_parameters.json


check whether you have "/data/dataset_folder/anomaly_scores.txt",
"/data/dataset_folder/concatenate_result.txt",
"/data/dataset_folder/explanations.txt" in the same directory as your input data file.

9. cd ..
10. cd xpacs_offline
11. change the xpacs_parameters.json file with parameters and dataset you have selected
   (refer to xpacs_offline/helper.md) for details
12. python xpacs.py xpacs_parameters.json

check whether you have "xpacs_offline/result/dataset_name/" directory with clusters and rules in .txt formats

13. cd ..
14. cd lookout
15. change the lookout_parameters.json file with parameters and dataset you have selected
   (refer to lookout/helper.md) for details
16. python lookout.py lookout_parameters.json begin_index, end_index
    the begin_index and end_index is the beginning clusters you want and ending clusters you want
    for example: python lookout.py lookout_parameters.json 2 4 

check whether you have "assets/dataet_name" with lookout plots

17. change the myapp_parameters.json
18. python myapp.py
19. open the browser and type into the addresses that myapp.py provided