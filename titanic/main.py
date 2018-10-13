from train_model import *
from score_model import *
import warnings


warnings.filterwarnings("ignore")


def main():
    full_data = merge_data()
    data = preprocess_data(full_data)
    data = feature_select(data)
    obj = train_test_data(data)
    train = obj[0]

    print(line, "Create validation data ...", line)
    X_train = split_validation(train)[0]
    X_valid = split_validation(train)[1]
    y_train = split_validation(train)[2]
    y_valid = split_validation(train)[3]

    print("X_train shape: ", X_train.shape)
    print("X_valid shape: ", X_valid.shape)
    print("y_train shape: ", y_train.shape)
    print("y_valid shape: ", y_valid.shape)

    print(arrow, "Success !", arrow)

    # fit model
    model = train_rf_model(X_train, y_train)

    # predict model
    prediction = predict(model, X_valid)
    print(X_valid.shape)
    print(prediction.shape)

    # accuracy
    accuracy_report(y_valid, prediction)

    # export pkl
    export_pkl(model, "random_forest_model")

    # load pkl
    load_from_pkl = load_model()

    # predict with test data
    X_test = obj[2] # test data with features which `survived` has been removed
    print(X_test.shape)
    prediction = prediction_using_pkl(load_from_pkl, X_test)
    export_pred_csv(prediction)

    print(star, "Finish data622 HW2.", star)


if __name__ == "__main__":
    main()