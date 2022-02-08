import pandas as pd
from sklearn.preprocessing import LabelEncoder


def test_function():
    print("Testing")

def load_train_data():
    train_raw = pd.read_csv("../data/train.csv", parse_dates=[0],
                dtype={
                    'StateHoliday': str,
                    # 'Store': int,
                    # 'DayOfWeek': int
                    })
    return train_raw

def add_store_info(train):
    """ Add the store info to sales data.

    Merges the store info on the train table and returns the combined table
    """
    # Load store info
    store_info = pd.read_csv("../data/store_info.csv")
    # Merge store info onto train data
    train = pd.merge(left=train, right=store_info, how='left', on='Store')
    return train

def process_data(train_raw, drop_null=True):
    """ Data Processing """
    train = train_raw.copy()
    train.loc[:, 'StateHoliday'] = train.loc[:, 'StateHoliday'].replace(to_replace='0', value='d')

    # Drop customers
    train = train.drop("Customers", axis=1)

    # Encode StateHoliday
    le = LabelEncoder()
    train.loc[:, 'StateHoliday'] = le.fit_transform(train.loc[:, 'StateHoliday'])

    # Drop all where sales are nan or 0
    train = train.dropna(axis=0, how='any', subset=['Sales'])
    train = train.loc[train.loc[:, 'Sales']!=0, :]

    # Drop all null value
    if drop_null:
        train = train.dropna(axis=0, how='any')

    return train
