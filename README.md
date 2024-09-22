# Prediction_of_Seizures_using_LSTM_and_Bi-LSTM_from_EEG_Data.
## Description
### Import necessary libraries
#### Pandas
Used for manipulation of data and analysis. It provides data structures like DataFrames to work with the structured data.
#### numpy as np:
This is a library used for numerical operations in Python. It gives support and help to arrays and matrices, with mathematical functions to operate on the data structures.
train_test_split:
A function from sklearn.model_selection that splits arrays or matrices into random train and test subsets. It's commonly used to create training and testing datasets for model evaluation.
classification_report:
A function from sklearn.metrics that builds a text report showing the main classification metrics (precision, recall, F1 score) for each class in the dataset.
confusion_matrix:
A function from sklearn.metrics that computes the confusion matrix to evaluate the accuracy of a classification. It shows the true vs. predicted classifications.
accuracy_score:
A function from sklearn.metrics that calculates the accuracy of a classification model by comparing the predicted labels to the true labels.
Sequential:
A class from tensorflow.keras.models that represents a linear stack of layers. It's used to create models layer by layer.
Bidirectional:
A wrapper for recurrent layers (like LSTM) in Keras that allows the layer to learn from the input sequence in both forward and backward directions, improving context understanding.
LSTM:
A layer type in Keras that implements Long Short-Term Memory (LSTM) networks, which are useful for modeling sequences, especially in time series or natural language processing.
Dropout:
A layer in Keras that randomly sets a fraction of input units to 0 during training, which helps prevent overfitting by ensuring that the model does not rely too heavily on any particular feature.
Flatten:
A layer in Keras that flattens the input, converting a multi-dimensional array into a one-dimensional array, typically used before the dense layer.
Dense:
A layer in Keras that implements a fully connected neural network layer, where each neuron in the layer receives input from all neurons in the previous layer.


## Group
**Group Number** - 6<br>
**Leader Name** - Ketan Singh Rautela<br>
**Members Name** - Vedanshi Rana, Hardik Singh, Gaurai Gupta, Aryan Parihar.
## Dataset 
The Dataset is collected from **UCI Machine Learning Repository**.

Link to original dataset (https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition).<br>

The dataset that is used in our project is a pre-processed and re-structured/reshaped and is very commonly dataset featuring Epileptic Seizure Detection.



