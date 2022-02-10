""" Script to create submission the file for Kaggle

Arguments:
----------
 - pipeline: str, default 'best'
    The name of the pipeline to load. All pipelines
    are stored under data/trained_piplines.
    If you pass 'best' or no value, the best
    pipeline (as stored in this script) is used.

"""
import argparse
import numpy as np
from pipeline import load_pipeline
from processing import load_holdout_data, process_data, add_store_info, add_week_month_info, add_beginning_end_month

# current_best_pipeline = 'random_forest_1'
current_best_pipeline = 'lightgbm_first'

if __name__ == '__main__':
    # Load the pipeline name from passed arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', default='best', nargs='?')
    args = parser.parse_args()

    name = args.pipeline

    if (name is None) or (name=='best'):
        name = current_best_pipeline

    # Load the pipeline
    pipeline = load_pipeline(name)

    # Load and process holdout data
    holdout_raw = load_holdout_data()
    # breakpoint()

    holdout_id = holdout_raw.iloc[:, 0] + 1
    # breakpoint()
    holdout_raw = holdout_raw.iloc[:, 1:]

    # holdout_raw = holdout_raw.drop('Id', axis=1)
    holdout = add_week_month_info(holdout_raw)
    holdout = add_beginning_end_month(holdout)
    holdout = process_data(holdout)
    holdout = add_store_info(holdout)

    # cols_to_drop = ['Open', 'StateHoliday', 'Assortment']
    # holdout = holdout.drop(cols_to_drop, axis=1)


    # Use the pipeline to predict on holdout
    print(f' - Predicting for pipeline "{name}"...')
    # holdout = pipeline.transform(holdout)
    y_pred = pipeline.predict(holdout)

    import pandas as pd
    result = pd.DataFrame({'Id': holdout_id, 'Sales': y_pred})
    # result = pd.DataFrame({'Id': np.arange(1,len(holdout)+1), 'Sales': y_pred})
    result.to_csv('../data/submission.csv', index=False)


    # # Save the prediction
    # submission_path = "../data/submission.csv"
    # print(f' - Saving submission file at data/submission.csv')
    # np.savetxt(submission_path, y_pred, delimiter=",")

    
    