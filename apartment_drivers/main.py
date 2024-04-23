import pandas as pd
import sklearn as sk
from apartment_drivers.data_processing.data_cleaning import CleanDate
from apartment_drivers.model_training.model_training import train_linear_regression

# training four different models based on the previous scripts
# we can run this by going to the terminal directing to the right directory (using cd-command) 
# and write poetry run python 'path to main.py'
# poetry run ml flow ui --> to se the results of the models

def main():
    """
    Main function to execute the data cleaning and model training process.
    """
    DF = CleanDate()
    
    train_linear_regression(DF.clean_data, 'price')
    train_linear_regression(DF.clean_data, 'price', random_state=200)
    train_linear_regression(DF.clean_data, 'price', random_state=60)
    train_linear_regression(DF.clean_data, 'price', test_size=.3)


if __name__ == '__main__':
    main()
