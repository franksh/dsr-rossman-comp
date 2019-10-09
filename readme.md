## Rossman Kaggle Mini-Competition

This mini competition is adapted from the Kaggle Rossman challenge.  Please refrain from looking at the challenge on Kaggle until after you have finished - this will allow you to get a true measurement of where you are at as a data scientist.

## Setup

```bash
#  during the competition run
python data.py

#  at test time run
python data.py --test 1
```

## Dataset

The dataset is made of two csvs:

```
#  store.csv
['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

#  train.csv
['Date', 'Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday']
```

The test period is from 2014-08-01 to 2015-07-31 - the test dataset is the same format as `train.csv`.

## Scoring Criteria

The competition is scored based on a composite of predictive accuracy and reproducibility.

## Predictive accuracy

The task is to predict the `Sales` of a given store on a given day.

Submissions are evaluated on the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

Zero sales days are ignored in scoring.

The team scores will be ranked - the highest score (lowest RMSPE) will receive a score of 10 for the scoring criteria section.

Each lower score (higher RMSPE) will receive a score of 10-(1 * number in ranking). If they are ranked second, score will be 10-2 = 8. 

## Reproducibility

The entire model should be completely reproducible - to score this the teacher will clone your repository and follow the instructions as per the readme.  All teams start out with a score of 10.  One point is deducted for each step not included in the repo.

## Advice

Commit early and often

Visualize early

Ensemble

Look at the predictions your model is getting wrong - can you engineer a feature for those samples?

A well tuned random forest should be the baseline
