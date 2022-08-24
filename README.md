# Knowledge-Graph-Embeddings-and-R-GCN
To run the link prediction for the FB15K dataset with KGE models, run the file ‘main.py’. It makes use of the Ampligraph library, so before running this file it is necessary to create a environment with python 1.7 and tensorflow version between 1.15.2 and 2.0. After creating this environment the ampligraph library can be installed using pip and then the file ‘main.py’ can be executed correctly.

To run the link prediction for the FB15K dataset with R-GCN model, run the file ‘rgcn_fbk15.py’. This model is created in pytorch so for this file, the python environment must be created with pytorch and pykeen library.

The file ‘kg_creation’ creates the knowledge graph from the industrial dataset and then implements the KGE creation model using ComplEx scoring function for the link prediction task. It also includes the XGBoost classifier to make use of the embeddings for classification of the measurement nodes. This file requires the same environment as that of the ‘main.py’

The RGCN implementation for node classification is present in the file ‘node_classify.py’. The graph and model for this task is build using the library Stellargraph and thus it requires an environment with python 3.8 and the latest tensorflow version. Then, the library Stellargraph can be installed using pip to run this file correctly.
The XGBoost classifier for the industrial dataset is implemented in the file ‘general_classifier.py’ which requires the latest tensorflow version and XGBoost to be installed using pip. 
