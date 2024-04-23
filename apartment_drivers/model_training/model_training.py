import mlflow

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# importing MLflow (after installing it in poetry 
# importing train_test_split from sklearn --> can split the data 
# imporing LinearRegression to predict the price 
# importing one-hot encoding because we have categorical variables, and the model cannot handle categorical variables



# seperates into x-matrix and y-vector variables (not an actual data split) 
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the data into features (X) and target (y).

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: A tuple containing the features (X) and target (y) arrays.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


# linear regression training (rough model)
def train_linear_regression(df, target_column="price", test_size=0.2, random_state=42):
    """
    Train a linear regression model on the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str, optional): The name of the target column. Defaults to "price".
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
    """
    # Split the data
    X, y = split_data(df, target_column=target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # One-Hot Encoding:
    encoder_one_hot = OneHotEncoder()
    X_train_one_hot = encoder_one_hot.fit_transform(X_train[['energy_l', 'varme', 'roof_type']])
# important to one-hot-encode before training the model, because...
# one-hot-encoding is preprossesing in some cases and a part of the modeling in others --> need to consider
# if we do it before or after the split. 
# creating a new X_train_one_hot dataset


    # Build linear regression model
    model_one_hot = LinearRegression()
    model_one_hot.fit(X_train_one_hot, y_train)


    # Evaluate model on the test set
    X_test_one_hot = encoder_one_hot.transform(X_test[['energy_l','varme', 'roof_type']]) # transforming the test-data to one-hot-encoding as well
    y_pred_one_hot = model_one_hot.predict(X_test_one_hot)
    mse = mean_squared_error(y_test, y_pred_one_hot)        


    # # Set our tracking server uri for logging --> if we havent created a server, then it will set it by itself 
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080/") 


    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Apartment Price Model")

    # Start an MLflow run
    with mlflow.start_run():

        # Log the loss metric
        mlflow.log_metric("mse", mse) # the metrics we want to log about the model. 

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic linear regression model for the apartment price dataset.")

        # Log the model
        model_info = mlflow.sklearn.log_model( # tracking the log of the model 
            sk_model=model_one_hot,
            artifact_path="apartment_price_model", # tracking artifact path
            input_example=X_train,
            registered_model_name="linear_regression_model",
        )
        # saves a lof of thing about the model. Logs these metrics about our model. 
