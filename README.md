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

## [CNN model classifying traffic signs](./traffic_signs_classification_using_CNN.ipynb)
(GTSRB, ~104,000 images, 43 classes) using preprocessed 32×32 pickled data from [Kaggle](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed?select=data2.pickle). Includes EDA on class distribution, channel order fixes, and a 3-block Conv2D architecture with BatchNorm and dropout.

**Results:** Test Accuracy 98.12% | Test Loss 0.1145 | Weighted F1 0.98
