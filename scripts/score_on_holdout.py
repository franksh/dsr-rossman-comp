""" Script to create submission the file for Kaggle

Arguments:
----------
 - pipeline: str, default 'best'
    The name of the pipeline to load. All pipelines
    are stored under data/trained_piplines.
    If you pass 'best' or no value, the best
    pipeline (as stored in this script) is used.

 - holdout_path: str
    The path to the holdout data file.

"""
import os.path
import argparse
import numpy as np
import pandas as pd
from pipeline import load_pipeline
from processing import process_data, add_store_info, add_week_month_info, add_beginning_end_month, metric

# current_best_pipeline = 'lightgbm_first'
current_best_pipeline = 'LGBM_hyperparam_optim'

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Score the model on the holdout set')
    parser.add_argument('--pipeline', default='best', nargs='?', type=str,
                        help='The name of the pipeline to load')
    parser.add_argument('--holdout_path', nargs='?', required=True, type=str,
                        help='The path to the holdout file')
    args = parser.parse_args()

    name = args.pipeline
    holdout_path = args.holdout_path

    if (name is None) or (name=='best'):
        name = current_best_pipeline

    # Check if holdout file exists
    assert os.path.isfile(holdout_path),\
        f"No file found at path:\n{holdout_path}"


    print(f' - Loading Holdout dataset at: {holdout_path}')
    holdout_raw = pd.read_csv(holdout_path,
                    parse_dates=['Date'],
                    # index_col=0,
                    usecols=['Date', 'Sales', 'Store', 'DayOfWeek',
                            'Customers', 'Open', 'Promo',
                            'StateHoliday', 'SchoolHoliday'],
                    dtype = {
                        'StateHoliday': str
                    })
    
    # Separate target and feature
    X = holdout_raw.copy(deep=True).drop(columns=["Sales"])
    y = holdout_raw.loc[:, "Sales"].values

    # Process features
    X = add_week_month_info(X)
    X = add_beginning_end_month(X)
    X = process_data(X)
    X = add_store_info(X)

    # Load the pipeline
    pipeline = load_pipeline(name)

    # Use the pipeline to predict on holdout
    print(f' - Predicting for pipeline "{name}"...')
    # holdout = pipeline.transform(holdout)
    y_pred = pipeline.predict(X)

    error = metric(y, y_pred)
    print('------------------\n')
    print(f'Summary')
    print(f'- Tested model:\t {name}')
    print(f'- Test data:\t {holdout_path}')
    print(f'Result:\n- RMSPE:\t{error:.4f}\n')


    # # Save the prediction
    # submission_path = "../data/submission.csv"
    # print(f' - Saving submission file at data/submission.csv')
    # np.savetxt(submission_path, y_pred, delimiter=",")
