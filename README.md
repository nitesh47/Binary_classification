# Binary_classification
The record type dataset is given. It has 12 dependent variable and a single target variable. 
Target variable consists of two classes 1 and 0. 
I addressed this problem as a binary classification problem. The classes are highly imbalanced. 
In order to balance the classes, I have used an oversampling technique.

About the feature, some of the features are not important. 
To extract the main features from the corpus, I have used ExtraTreeClassifier and applied PCA to perform dimensionality reduction to avoid the curse of dimensionality problem.

After that, the dependable features are given to the neural network model and predictive analysis performed.


