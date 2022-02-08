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

def process_data(train_raw):
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
    train = train.dropna(axis=0, how='any')

    return train
