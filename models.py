from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Split the complete training set in a train and test set for training
def trainTestSplit(X_Train, Y_Train):
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=0)
    return x_train_split, x_test_split, y_train_split, y_test_split

# Evaluate the model on the test set taken from the training data
def getTrainingAccuracy(model, x_te, y_te):
    # Make predictions on the testing data
    y_pred = model.predict(x_te)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_te, y_pred)
    # Print the name of the model and its accuracy on the training data
    print(type(model).__name__ + " \tAccuracy: ", accuracy)

# Create the XGBoost classifier 
def XGBoost(x_tr, y_tr, plot_importance=False):
    model = XGBClassifier() #random_state=1, eta = 0.1, gamma = 0.5, max_depth = 7
    # Fit the model on the training data
    model.fit(x_tr, y_tr)
    
    # When true this plots the model feature importance plot
    if plot_importance:
        # Get feature importance scores
        feature_importance = model.feature_importances_

        # Create a bar plot to visualize feature importance
        plt.figure(figsize=(25, 6))
        plt.bar(x_tr.columns, feature_importance)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance')
        plt.xticks(rotation=90, fontsize=10, ha='center')
        plt.show()
    
    return model