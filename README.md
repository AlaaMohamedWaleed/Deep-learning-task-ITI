# Deep-learning-task-ITI
## [Customer Churn Classification (FCNN)](./Classification_using_FCNN_on_Churn_Modelling_Dataset.ipynb)

Binary classification model predicting customer churn using a Fully Connected Neural 
Network on the Churn Modelling dataset (10,000 customers, ~80/20 imbalance). Includes 
EDA, preprocessing, class-weighted training, and threshold tuning

**Results:** Accuracy 0.85 | F1 (churn) 0.62 | ROC-AUC 0.86

## [Bengaluru House Price Prediction (FCNN)](./Regression_using_FCNN_on_bengaluru_house_prices_dataset.ipynb)

Regression model predicting house prices using a Fully Connected Neural Network on the 
Bengaluru house price dataset (~13,000 listings). Includes cleaning of messy fields 
(mixed-unit sqft, corrupted bedroom counts, high-cardinality location), price-per-sqft 
based outlier validation, and learning rate tuning.

**Results:** MAE 32.35 lakhs | RMSE 66.55 lakhs | R² 0.730
