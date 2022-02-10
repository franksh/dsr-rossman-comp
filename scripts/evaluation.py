"""
Methods to evaluate the model with
"""
import numpy as np
import pandas as pd

from .processing import load_train_data, process_data

def metric_not_summed(preds, actuals):
    """ The RMSPE metric per observation """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    # return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
    # return 100 * np.abs((actuals - preds) / actuals)
    return 100 * (actuals - preds) / actuals


def compare_predictions_and_sales_per_date(y_pred, y_actual):
    """ This method creates a dataframe with actual and predicted sales

    Parameters:
    -----------
     - y_pred: np.array
        Contains the predicted values of sales
     - y_actual: np.array
        Contains the actual values of sales

    Returns:
    --------
     - sales: pd.DataFrame
        A table containing the error between predicted and actual,
        where the error is the 
        with columns:
            Date | pred | actual | error | error_abs 
    """

    assert len(y_pred)==len(y_actual), "y_pred and y_actual differ in length"

    # Load the dates
    train_raw = load_train_data()
    dates = pd.DataFrame(process_data(train_raw, drop_null=True, drop_date=False).loc[:, 'Date'])

    # Merge predicted and actual values
    sales = dates.copy()
    sales.loc[:, 'pred'] = y_pred
    sales.loc[:, 'actual'] = y_actual

    # Sum up sales by date
    sales = sales.groupby(by='Date').sum().reset_index()

    # sales.loc[:,'weekday'] = sales.loc[:, 'Date'].dt.weekday

    # Calculate the error
    sales.loc[:, 'error'] = metric_not_summed(sales.loc[:, 'pred'].values, sales.loc[:, 'actual'].values)
    sales.loc[:, 'error_abs'] = np.abs(sales.loc[:, 'error'])
    return sales