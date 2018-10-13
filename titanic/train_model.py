# This file include functions that performs data pre-processing, fitting a random forest model
# and exporting the model as .pkl file in local directory.
from pull_data import read_train, read_test, join_path
from cf_report import csv_cf_report, plot_cf_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

line = "=" * 20
arrow = ">" * 20
star = "*" * 20


def merge_data():
    # combine the train and test dataset
    print(line, "Reading train data...", line)
    train = read_train()

    print(line, "Reading test data...", line)
    test = read_test()

    train['label'] = "train"
    test['label'] = "test"

    print(line, "Merging train, test data...", line)
    full_data = train.append(pd.DataFrame(data=test), ignore_index=True, sort=True)
    print("Merged data shape: ", full_data.shape)
    print("Merged data columns: ", full_data.columns.values)
    print(line, "Merging Done!", line)
    print(arrow, "Success !", arrow)
    return full_data
###################################################################################################


def get_title(name):
    """
    This function extracts the title from name column in data.
    The surname positions before `,` and the title positions between `,` and `.`.
    First name places behind `.`. We simply split the name to extract title.
    """
    # Extract title from Name column
    return name.split(", ")[1].split('. ')[0]


def preprocess_data(data):
    """
    The preprocessing performs cleaning, imputation and feature engineering.
    All values drawn from merged data.

    Age: We create Age_Filled column to fill in median age calculated grouped by pclass and sex
        and fills in accordingly then drop the column and rename it as Age.
    Age_group: We create Age group divided by groups as we binned in EDA.
    Fare: Fill in the missing value in Fare column for median value.
    Fare_group: We create Fare group divided by groups as we binned in EDA.
    Cabin_exist: Cabin data is very sparse. Assuming this column was filled for after survival,
        we simply create a column that represents whether the row contains cabin information.
    Title: We extract title from person's name and simplify by combining the same type of title.
    Family_size: We add SibSp(sibling, spouse) + Parch(parents, child) + myself as the number of family size.
        Furthermore, we create dummy variables for family size in groups (big,mideum,small) and female/mother/child.

    """
    print(line, "Data Preprocessing ...", line)
    # fill in the missing values in age column with median grouped by Pclass and Sex
    data['Age_filled'] = data['Age']
    median_age = data.groupby(['Pclass', 'Sex'])['Age'].median()

    data.loc[(data['Pclass'] == 1) & (data['Sex'] == 'female') & (data['Age'].isnull()), 'Age_filled'] = median_age[1][0]
    data.loc[(data['Pclass'] == 2) & (data['Sex'] == 'female') & (data['Age'].isnull()), 'Age_filled'] = median_age[2][0]
    data.loc[(data['Pclass'] == 3) & (data['Sex'] == 'female') & (data['Age'].isnull()), 'Age_filled'] = median_age[3][0]

    data.loc[(data['Pclass'] == 1) & (data['Sex'] == 'male') & (data['Age'].isnull()), 'Age_filled'] = median_age[1][1]
    data.loc[(data['Pclass'] == 2) & (data['Sex'] == 'male') & (data['Age'].isnull()), 'Age_filled'] = median_age[2][1]
    data.loc[(data['Pclass'] == 3) & (data['Sex'] == 'male') & (data['Age'].isnull()), 'Age_filled'] = median_age[3][1]

    data.drop('Age', axis=1, inplace=True)
    data.rename(columns={'Age_filled': 'Age'}, inplace=True)

    # create age group
    bins = [0, 16, 32, 48, 64, 81]
    labels = [0, 1, 2, 3, 4]
    data['Age_group'] = pd.cut(data['Age'], bins=bins, labels=labels, include_lowest=True)

    # fill in one missing value in Fare column
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # create age group
    bins = [0, 7.91, 14.45, 31, 515]
    labels = [0, 1, 2, 3]
    data['Fare_group'] = pd.cut(data['Fare'], bins=bins, labels=labels, include_lowest=True)

    data['Cabin_exist'] = data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    # Extract title from name and simplify the titles
    data['Title'] = data['Name'].apply(get_title)
    data['Title'].replace('Mlle', 'Miss', inplace=True)
    data['Title'].replace('Ms', 'Miss', inplace=True)
    data['Title'].replace('Mme', 'Mrs', inplace=True)
    data['Title'].replace(['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major','Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others', inplace=True)

    # Get family size in numbers
    data['Family_size'] = data["SibSp"] + data["Parch"] + 1

    # create family groups, mother, female and child
    data['Child'] = np.where(data['Age'] < 18, 1, 0)
    data['Mother'] = np.where((data['Sex'] == "female") & (data['Parch'] > 0) & (data['Child'] == 1), 1, 0)
    data['Female'] = np.where((data['Sex'] == "female"), 1, 0)
    data.loc[data['Family_size'] == 1, 'Family_type'] = 'Single'
    data.loc[(data['Family_size'] > 1) & (data['Family_size'] < 5), 'Family_type'] = 'Medium'
    data.loc[data['Family_size'] >= 5, 'Family_type'] = 'Big'

    # create dummy variables for pclass, embarked, family_type, title and concatenate the columns
    Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
    Family = pd.get_dummies(data['Family_type'], prefix='Family')
    Title = pd.get_dummies(data['Title'])
    data = pd.concat([data, Family, Pclass, Embarked, Title], axis=1)

    print("preprocessed data shape: ", data.shape)
    print(arrow, "Success !", arrow)
    return data
###################################################################################################


def feature_select(data):
    print(line, "Selecting features ...", line)
    # We manually type in the features instead of dropping the original columns.
    # This is to leave the feature engineering possibilities
    features = ['Pclass', 'Survived', 'label', 'Age', 'Age_group', 'Fare', 'Fare_group',
                'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                'Family_size', 'Family_Big', 'Family_Medium', 'Family_Single', 'Child', 'Mother',
                'Master', 'Miss', 'Mr', 'Mrs', 'Others', 'Female', 'Cabin_exist']
    print("The number of features w/ label and Survived column: ", len(features) - 2)
    data = data[features]
    print(arrow, "Success !", arrow)
    return data


def train_test_data(data):
    # When we merged the data for better imputation result, we labeled them as train/test.
    # Now we split the data in order to revert them as it was, and drop Survived column in test data.
    print(line, "Splitting data into train, test ...", line)
    train = data.loc[data['label'] == 'train']
    test = data.loc[data['label'] == 'test']
    X_test = test.drop(['label', 'Survived'], axis=1)
    print("train data shape: ", train.shape)
    print("test data shape: ", test.shape)
    print(arrow, "Success !", arrow)
    return train, test, X_test


def split_validation(train):
    # As we are handling small size of dataset, it is a good idea to create validation dataset within the train dataset,
    # to avoid over/under-fitting situations.
    y_train = train['Survived']
    X_train = train.drop(['Survived', 'label'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=23)
    return X_train, X_valid, y_train, y_valid

    #print("X_train shape: ", X_train.shape)
    #print("X_valid shape: ", X_valid.shape)
    #print("y_train shape: ", y_train.shape)
    #print("y_valid shape: ", y_valid.shape)
###################################################################################################


# Fit model using validation-training data
def train_rf_model(X_train, y_train):
    # The random forest model parameters are results from EDA.
    print(line, "Training random forest model using validation-training data ..", line)
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                   max_depth=13, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=2, min_samples_split=10,
                                   min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
                                   oob_score=False, random_state=None, verbose=0,
                                   warm_start=False)
    model.fit(X_train, y_train)
    print(arrow, "Success !", arrow)
    return model


# predict
def predict(model, X_valid):
    # Make predictions on the validation dataset using the trained random forest model above.
    print(line, "Predicting using random forest model ... ", line)
    prediction = model.predict(X_valid)
    print(arrow, "Success !", arrow)
    return prediction


# measure accuracy using prediction and y_valid data
def accuracy_report(y_valid, prediction):
    # Create accuracy report, plot them and print accuracy score.
    target_names = ['Perished', 'Survived']
    report = classification_report(y_valid, prediction, target_names=target_names)
    print(line, "Accuracy Score", line)
    score = accuracy_score(y_valid, prediction)
    print("Accuracy Score = {0:5f}".format(score))
    print(line, "Generate classification report and export it as .csv", line)
    csv_cf_report(report)
    print(arrow, "Success !", arrow)

    print(line, "Plot the classification report and save image as .png", line)
    plot_cf_report(report)
    print(arrow, "Success !", arrow)
###################################################################################################


def export_pkl(model, file_name):
    print(line, 'Save RandomForest model as', file_name + '.pkl', line)
    # Dump the trained random forest classifier with Pickle
    filename = file_name + '.pkl'
    rf_pkl_file = join_path('model', filename)
    # Open the file to save the pkl file
    rf_model_pkl = open(rf_pkl_file, 'wb')
    pickle.dump(model, rf_model_pkl)
    # Close the pickle instances
    rf_model_pkl.close()
    print(arrow, "Success !", arrow)