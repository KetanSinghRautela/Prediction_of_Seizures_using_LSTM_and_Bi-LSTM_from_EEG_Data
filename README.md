# Prediction_of_Seizures_using_LSTM_and_Bi-LSTM_from_EEG_Data.
## Description
### Import necessary libraries:-
- **pandas as pd-**
Used for manipulation of data and analysis. It provides data structures like DataFrames to work with the structured data.<br>
- **numpy as np-**
This is a library used for numerical operations in Python. It gives support and help to arrays and matrices, with mathematical functions to operate on the data structures.<br>
- **train_test_split-**
It is a function from sklearn.model_selection that splits the arrays or matrices into the random training and testing subsets. It's basically used to create training and testing datasets for model evaluation ahead.<br>
- **classification_report-**
This is  function from sklearn.metrics which builds text report that shows the main classification metrics (precision, recall, F1 score) that are calculated for the overall results of the model for each class in the dataset.<br>
- **confusion_matrix-**
This is also a function from sklearn.metrics which computes the confusion matrix to evaluate the accuracy of the classification. It tells the true vs. predicted classifications.<br>
**accuracy_score-**
Also a function from sklearn.metrics which calculates the accuracy of the classification model by comparing it with the predicted labels to true labels.<br>
- **Sequential-**
It is a class from tensorflow.keras.models that represents linear stack of layers. It is used to create the models layer by layer.<br>
- **Bidirectional-**
This is a wrapper for Recurrent layers (like LSTM in our project) in Keras which allows the layer to learn from the input sequence in both forward and backward directions, it improves the understanding of the context.
- **LSTM-**
This is a layer type in Keras which implements Long Short-Term Memory (LSTM) networks, that are useful for modeling sequences, generally used  in time series or natural language processing.
- **Dropout-**
This is a layer in Keras that randomly sets a fraction of input units to 0 while training, that prevents overfitting by ensuring that our model does not rely heavily on a particular feature.
- **Flatten-**
It is a layer in Keras that flattens the input provided, converts a multi-dimensional array to a 1-D array,  used before dense layer.
- **Dense-**
It it a layer in Keras that implements fully connected Neural Network layer, where every neuron of the layer receives input from all the neurons in previous layer.<br>
### Load the Dataset
- **Read Data-** The dataset is loaded from a .CSV file using pandas. This contains EEG recordings and associated labels indicating whether a seizure occurred.
- **Structure of Data-** Dataset  consists of EEG readings and a target label that indicates the seizure's presence or absence.

## Group
**Group Number** - 6<br>
**Leader Name** - Ketan Singh Rautela<br>
**Members Name** - Vedanshi Rana, Hardik Singh, Gaurai Gupta, Aryan Parihar.
## Dataset 
The Dataset is collected from **UCI Machine Learning Repository**.

Link to original dataset (https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition).<br>

The dataset that is used in our project is a pre-processed and re-structured/reshaped and is very commonly dataset featuring Epileptic Seizure Detection.



