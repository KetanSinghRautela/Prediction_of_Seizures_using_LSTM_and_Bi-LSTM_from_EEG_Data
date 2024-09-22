# Prediction_of_Seizures_using_LSTM_and_Bi-LSTM_from_EEG_Data.
# Description
Epilepsy is a long-term condition that significantly affects both individuals and society in terms of health and social implications.  It is marked by repeated seizures, which are short episodes of uncontrollable movements that can affect either a specific part of the body or the entire body.  Approximately 50 million people around the world are affected by epilepsy, making it one of the most prevalent neurological disorders. Despite the availability of treatments and interventions to manage the condition, a significant number of patients still experience epileptic seizures and are not fully controlled.<br><br>
This project aims to develop a Machine Learning Model that is capable of predicting Epileptic Seizures from EEG (electroencephalogram) data/recordings using a Bidirectional Long Short-Term Memory (Bi-LSTM) network and (LSTM). EEG signals provide insights into the brains activity, accurately predicting seizures can improve the quality of life for patients with epilepsy.A trained Bi-LSTM model is slightly more accurate than the LSTM model in predicting seizures from EEG data. Results are shown in the OUTPUT files for both the models.<br><br>
For the future work and enhancing the model you can
- Increase the number of epochs.
- Use a dataset(more diverse EEG recordings) which is of larger size than the one used in this project.
- Explore the hyperparameter tuning to enhance model performance.
- Implement additional features, like real-time prediction and integration with some kind of wearable devices.







# Workflow
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
- **Structure of Data-** Dataset  consists of EEG readings and a target label that indicates the seizure's presence or absence.<br>
### Preprocessing Data
- **Separate Features and Labels-**The target variable y (labels) that we declared is extracted from DataFrame. This indicates whether the seizure occurred (1 if yes) or not (0 if not occured). Unnecessary columns are dropped which includes any non-predictive columns, like identifiers.
- **Reshaping the Input Data-**
The features are then converted into a specific shape(reshaped) that is suitable for applying LSTM. Each of the feature row is now converted into a 3-D Array format. This is necessary as LSTM expects input in the form of (samples, timesteps, features).Each EEG reading is then structured into smaller arrays for LSTM processing.<br>
### Binary Classification Adjustment
- **Binary Encoding-** Labels are adjusted to the binary format where a label other than 1 is set to 0. It simplifies our classification to binary seizure detection.<br>
### Split the Data
- **Train-Test Split-** Dataset is splitted into training and testing sets using train_test_split. This helps in evaluating the model's performance on unseen data.<br>
### Building the LSTM/Bi-LSTM Model
- **Sequential Model-** A sequential model is initialized to build a LSTM architecture.
- **Bidirectional-LSTM Layers-** Bi-LSTMs are added to the model. The LSTM now processes the input data in both forward and backward directions, capturing more temporal dependencies.<br>
### Compile the Model
Model is compiled with a binary Cross-Entropy Loss function and the Adam optimizer. This is the setup that's suitable for binary classification tasks.<br>
### Train the Model
Model is now trained using the .fit method. During training, it learns to recognize the patterns related to seizures based on  input EEG data.<br>
### Evaluate the Model
After training, model is now evaluated on the test set to determine performance.<br>
## Group
**Group Number** - 6<br>
**Leader Name** - Ketan Singh Rautela<br>
**Members Name** - Vedanshi Rana, Hardik Singh, Gaurai Gupta, Aryan Parihar.
## Dataset 
The Dataset is collected from **UCI Machine Learning Repository**.

Link to original dataset (https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition).<br>

The dataset that is used in our project is a pre-processed and re-structured/reshaped and is very commonly dataset featuring Epileptic Seizure Detection.
### Results
At last, accuracy of the model is printed, provids an indication of its effectiveness in predicting seizures based on EEG data.
- **LSTM-** When trained with **LSTM** model, the accuracy that we get is<br><img width="871" alt="Accuracy" src="https://github.com/user-attachments/assets/5dd88305-9a7d-4550-b3f8-2269585eaac5"><br>
- **Bi-LSTM-** When trained with **Bi-LSTM** model, the accuracy that we get is<br><img width="901" alt="Bi-LSTM_Accuracy" src="https://github.com/user-attachments/assets/4b6cd635-7a1b-409f-8f12-7b1e2710618b">





