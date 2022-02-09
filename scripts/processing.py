import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_train_data():
    train_raw = pd.read_csv("../data/train.csv",
                    parse_dates=[0],
                    dtype = {
                        'StateHoliday': str,
                    })
    return train_raw

def load_holdout_data():
    holdout_raw = pd.read_csv("../data/holdout_b29.csv",
                    parse_dates=[1],
                    # index_col=0,
                    dtype = {
                        'StateHoliday': str
                    })
    return holdout_raw

def add_store_info(train):
    """ Add the store info to sales data.

    Merges the store info on the train table and returns the combined table
    """
    # Load store info
    store_info = pd.read_csv("../data/store_info.csv")
    # Merge store info onto train data
    train = pd.merge(left=train, right=store_info, how='left', on=['Store', 'month'])
    return train

def add_week_month_info(train):
    """
    Add week and month information as another column to the features
    """
    
    train.loc[:,'week'] = train.loc[:,'Date'].dt.isocalendar().week.astype(int)
    train.loc[:,'month'] = train.loc[:,'Date'].dt.month
    return train

def add_beginning_end_month(train):
    """
    Add features that represent the beginning and end of months
    """
    def get_feature_end_month(day_of_month):
        return (day_of_month/31)**4

    def get_feature_beginning_month(day_of_month):
        return ((31-day_of_month)/31)**4

    # get_feature_end_month(33)
    train.loc[:, 'end_of_month'] = train.loc[:, 'Date'].dt.day.apply(get_feature_end_month)
    train.loc[:, 'beginning_of_month'] = train.loc[:, 'Date'].dt.day.apply(get_feature_beginning_month)
    return train
        
def process_data(train_raw, drop_null=True, drop_date=True):
    """ Data Processing """
    train = train_raw.copy()
    train.loc[:, 'StateHoliday'] = train.loc[:, 'StateHoliday'].replace(to_replace='0', value='d')

    # Drop customers and Date
    train = train.drop("Customers", axis=1)
    if drop_date:
        train = train.drop("Date", axis=1)

    # Drop all where sales are nan or 0
    if 'Sales' in train.columns:
        train = train.dropna(axis=0, how='any', subset=['Sales'])
        train = train.loc[train.loc[:, 'Sales']!=0, :]

    # Drop all null value
    if drop_null:
        train = train.dropna(axis=0, how='any')

    return train


def metric(preds, actuals):    
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])