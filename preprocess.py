import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt

from models import *

def preprocessingPipeline(input_df):
    # Drop columns that are not usefull
    input_df = input_df.drop(['Ticket'], axis = 1)
    input_df = input_df.drop(['Name'], axis = 1)
    input_df = input_df.drop(['Cabin'], axis = 1)

    # Replace all NAN cabin values with a custom value
    input_df = input_df.fillna({'Cabin': 'Z999'})

    # Replace all missing age values with the mean age
    mean_age = input_df['Age'].mean()
    input_df = input_df.fillna({'Age': mean_age})

    # Replace all missing Embarked values with S as its the most common value
    input_df = input_df.fillna({'Embarked': 'S'})

    # List all the categorical variables
    categorical_columns = ['Pclass','Sex','Embarked']

    # Create dummy column with one hot encoding for all categorical variables
    input_df = pd.get_dummies(input_df,columns=categorical_columns, dummy_na=False)

    target_df = []
    # If there is a target column (then its the train set) then Seperate the target column and drop it from the train DF
    if 'Survived' in input_df:
        target_df = input_df['Survived']
        input_df = input_df.drop(['Survived'], axis = 1)
        # Only drop the passengerId if it is the training set not the test set
        input_df = input_df.drop(['PassengerId'], axis = 1)

    # Transform all columns to float type
    for column in input_df.columns:
        if column != 'PassengerId':
            input_df[column] = input_df[column].astype(float) 

    return input_df, target_df

# Predict the labels on the test dataset for submission
def predictOnTestSet(X_Test, model):
    # Seperate the column PassengerId from the X_Test df
    PassengerId_Col = X_Test['PassengerId']
    X_Test = X_Test.drop(['PassengerId'], axis = 1)

    predictions = model.predict(X_Test)
    # convert the range of 0 to 1 output to either true or false
    predicted_labels = predictions > 0.5

    # Create a DataFrame with PassengerId and Transported columns
    output_df = pd.DataFrame({
        'PassengerId': PassengerId_Col,
        'Survived': predicted_labels
    })

    # Convert boolean values to 'True'/'False' strings
    output_df['Survived'] = output_df['Survived'].map({True: 1, False: 0})

    # Save the result to a CSV file
    output_df.to_csv('predictions.csv', index=False)
    

if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\JussiBoersma\Documents\python_practice\AI\Kaggle\Titanic\data\train.csv")

    # Age has 177 NA values
    # Cabin has 687 NA values
    # Embarked has 2 NA values

    X_Train, Y_Train = preprocessingPipeline(df)

    x_tr, x_te, y_tr, y_te = trainTestSplit(X_Train, Y_Train)

    model = XGBoost(x_tr, y_tr, plot_importance=False)

    getTrainingAccuracy(model, x_te, y_te)

    # Evaluate the model on the test set
    df = pd.read_csv(r"C:\Users\JussiBoersma\Documents\python_practice\AI\Kaggle\Titanic\data\test.csv")
    X_Test, _ = preprocessingPipeline(df)

    predictOnTestSet(X_Test, model) 
 