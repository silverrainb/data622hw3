# this file will load the model from randomforest.pkl file in local directory
# and export predictions as .csv file.
import pickle
import pandas as pd
from pull_data import join_path
line = "=" * 20
arrow = ">" * 20


# Load the random forest model pickle file
def load_model():
    random_forest_model_pkl = open(join_path("model", "random_forest_model.pkl"), 'rb')
    random_forest_model = pickle.load(random_forest_model_pkl)
    print("Loaded Random Forest Model: ", random_forest_model)
    return random_forest_model


# Predict using the model and X_test dataset
def prediction_using_pkl(model, X_test):
    print(line, "Predicting using random forest model loaded from .pkl ... ", line)
    prediction = model.predict(X_test)
    print(arrow, "Success !", arrow)
    return prediction


# Export predictions as .csv file
def export_pred_csv(prediction):
    print(line, "export prediction as predicted_score.csv in local directory", line)
    submission = pd.read_csv(join_path("data", "prediction_template.csv"), index_col="PassengerId")
    submission["Survived"] = prediction
    print(submission.shape)
    submission.head()
    submission.to_csv(join_path("data", "random_forest_submission.csv"))
    print(arrow, "Success !", arrow)