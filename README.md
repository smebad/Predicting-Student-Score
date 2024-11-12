# Student Performance Prediction 📊🎓

**Author:** [Syed Muhammad Ebad](https://www.kaggle.com/syedmuhammadebad)  
**Date:** 30-Oct-2024  
[Send me an email](mailto:mohammadebad1@hotmail.com)  
[Visit my GitHub profile](https://github.com/smebad)

---

## 🚀 Overview

In this project, we aim to **predict students' average scores** based on various features such as **gender**, **parental level of education**, **lunch type**, **test preparation course**, and their individual subject scores in **math**, **reading**, and **writing**. Using a machine learning model, we analyze the relationships between these features and predict how well a student will perform.

---

## 📊 Dataset

The dataset used for this project is the **Students Performance Dataset**, which includes data on students' scores in three subjects (Math, Reading, and Writing) and several other demographic features such as:

- **gender**
- **race/ethnicity**
- **parental level of education**
- **lunch**
- **test preparation course**

You can find the dataset [here](https://www.kaggle.com/datasets/muhammadroshaanriaz/students-performance-dataset-cleaned/data).

---

## 🧠 What We Did

1. **Data Loading & Exploration**:
   - We loaded the cleaned dataset and explored its structure using `.info()` and `.describe()` to understand the data better.
   - We performed an **Exploratory Data Analysis (EDA)** to visualize important patterns such as:
     - Distribution of student grades
     - Feature correlations
     - Impact of parental education on student performance

2. **Data Preprocessing**:
   - Split the data into features (`X`) and target variable (`y`).
   - Split the data into training and test sets.
   - Applied **Standard Scaling** to numeric features and **One-Hot Encoding** to categorical features for better model performance.

3. **Model Selection & Training**:
   - We chose the **Random Forest Regressor** due to its excellent performance with both numeric and categorical data.
   - Created a pipeline with preprocessing and the regression model.
   - Trained the model on the training set and evaluated it on the test set.

4. **Evaluation**:
   - The model was evaluated using several metrics:
     - **Mean Absolute Error (MAE):** 0.53
     - **Mean Squared Error (MSE):** 1.31
     - **Root Mean Squared Error (RMSE):** 1.15
     - **R-squared (R²):** 0.99
   - The model performed exceptionally well, achieving a very high **R² score** of 0.99, indicating that the model explains 99% of the variance in the target variable.

5. **Making Predictions**:
   - We used the trained model to predict the average score for new students based on their features.
   - A sample prediction showed the model's ability to predict student scores accurately.

---

## 🔍 What We Learned

- **Data Exploration**: Gaining insights into the dataset through visualizations and statistical summaries was crucial for understanding the relationships between different features and the target variable.
- **Feature Engineering**: The importance of properly preprocessing categorical and numerical features cannot be overstated. Scaling numeric features and encoding categorical variables helped improve model performance.
- **Model Performance**: The Random Forest model worked well for this regression task, providing highly accurate predictions with minimal error.
- **Evaluation Metrics**: By using RMSE, MSE, MAE, and R², we were able to evaluate how well our model was performing and ensure that it was not overfitting or underfitting.

---

## 📈 Evaluation Results

Here are the evaluation results from the Random Forest Regressor model:

- **MAE (Mean Absolute Error):** 0.53  
  The average error between predicted and actual scores is less than 1 point, which indicates the model is quite accurate.
  
- **MSE (Mean Squared Error):** 1.31  
  This small value suggests that the model is making accurate predictions with minimal error.
  
- **RMSE (Root Mean Squared Error):** 1.15  
  On average, the model's predictions are off by 1.15 points, which is very low, indicating a good model fit.
  
- **R² (R-squared):** 0.99  
  An excellent R² score, suggesting that the model explains 99% of the variance in student performance.

---

## 💡 Conclusion

This project demonstrates how machine learning can be used to predict student performance based on various factors. The Random Forest model performed exceptionally well, with minimal error and high accuracy. By applying robust data preprocessing and feature engineering techniques, we were able to create a reliable predictive model.

The key takeaway is that **model evaluation metrics** such as **RMSE**, **MSE**, **MAE**, and **R²** provide a comprehensive understanding of model performance and guide improvements where necessary.

---

## 🛠️ Technologies Used

- **Python**: For building the machine learning model and performing data analysis.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning, data preprocessing, and model evaluation.
- **Jupyter Notebook**: For running and documenting the project.

---

## 📅 Next Steps

- **Model Tuning**: We could explore hyperparameter tuning to further improve the model's performance using techniques like **GridSearchCV**.
- **Other Models**: Testing other models such as **Linear Regression** or **Gradient Boosting Machines (GBM)** could provide insights into which model works best for this problem.
- **Additional Features**: Incorporating more features, such as study hours or extracurricular activities, may improve prediction accuracy.

---

## 👏 Acknowledgments

- **Dataset Source**: The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/muhammadroshaanriaz/students-performance-dataset-cleaned/data).
- **Libraries Used**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.

---

### If you like this project, dont forget ot give a star on this repo! ⭐

