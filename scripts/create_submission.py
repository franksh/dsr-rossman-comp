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
from processing import load_holdout_data, process_data, add_store_info, add_week_month_info

current_best_pipeline = 'random_forest_1'

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
    holdout = add_week_month_info(holdout_raw)
    holdout = process_data(holdout)
    holdout = add_store_info(holdout)

    # Use the pipeline to predict on holdout
    print(f' - Predicting for pipeline "{name}"...')
    y_pred = pipeline.predict(holdout)

    # Save the prediction
    submission_path = "../data/submission.csv"
    print(f' - Saving submission file at data/submission.csv')
    np.savetxt(submission_path, y_pred, delimiter=",")

    
    