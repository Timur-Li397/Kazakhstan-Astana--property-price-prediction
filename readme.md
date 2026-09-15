Astana Property Price Prediction

This is a machine learning project aimed at predicting the asking prices of apartments in Astana, Kazakhstan.
  
  
  Overview

The project uses 18,388 real estate listings to predict apartment prices based on property features like size, number of rooms, location and other details.
Workflow:
- Data cleaning and exploratory data analysis
- Feature preprocessing
- Linear Regression model
- Random Forest model
- Model comparison
- Feature importance analysis
- Building a Streamlit prediction app



 Results

| Model | MAE (KZT) | RMSE (KZT) | R² |
|---|---:|---:|---:|
| Linear Regression | 8.00M | 14.78M 0.8496 |
| Random Forest | 6.91M | 12.93M *0.8849** |
The Random Forest model gave better results compared to the Linear Regression model. It had MAE and RMSE values and a higher R² score meaning it was more accurate, in predicting apartment prices.



 Technologies


Python · Pandas · Scikit-learn · Streamlit · Matplotlib · Git · GitHub




The model predicts asking prices, not final transaction prices. Extreme high-value properties may have larger prediction errors.

Author

Timur Li
Data Science Student, City University of Hong Kong