# Time-Series-Forecasting
Using ML To Forecast Walmart Weekly Sales Using Scikit-learn And Lag Features

## Objectives
- Data Inspection/Merging
- Sales Visualization
- Date-Time Extraction
- Lag Feature Creation
- Feature Selection/Engineering
- Model Creation
- Model Evaluation
- Actual and Predicted Sales Visualization
- Feature Importance
- Next Few Weeks Forecasting and Visualization
- Streamlit Deployment

## Tools & Library Used
- Python
- Pandas
- numpy
- matplotlib
- Scikit-learn
- Joblib

## ML Workflow
- Data Preprocessing
- Lag Feature Creation
- Feature Selection
- Model Creation/Training
- Model Evaluation
- Next 4-Weeks Forecast
- Feature Importance Extraction

## Findings
More n_estimators for small dataset wont help increase r2 percentage and more lag features doesnt guarantee higher predictive accuracy

Week dominate in feature importance due to some certain reasons
- holiday weeks
- Black Friday periods
- year-end shopping weeks
predicted sales doesnt really aligned well with actual sales but At times knowing when to stop improving model is also an ML skill.

## Conclusion
So after all the tweeking and all the feature engineering and feature selections, i later had to use 20 n_estimators instead of 100 because of the reduction in dataset and i only use lag 1,2 and 4 and week as feature because that have better feature importance that helps increase percentage of r2(Accuracy).

## Challenges
Wrong Lag Creation -
- I initially created lag features globally across all stores, which mixed different store sales together and reduced model performance.

✅ Lesson:
Lag features in forecasting should usually be created within the same time-series group.

Leakage Risk
- Rolling average initially used current sales information.

✅ Lesson:
Forecasting features should only use past information to avoid data leakage.

Small Dataset Limitation

- Aggregating data reduced dataset size significantly, making Random Forest unstable sometimes.

✅ Lesson:
Model choice depends heavily on dataset size and structure.

Chasing Metrics Too Much

- Adding many features did not always improve performance.

✅ Lesson:

More features do not always mean better models.

## Author
Amos Kolawole
