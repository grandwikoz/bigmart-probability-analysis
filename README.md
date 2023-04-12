# bigmart-probability-analysis
A project for Pacmann's Probability Course

# Background
This data comes from Kaggle (https://www.kaggle.com/datasets/thedevastator/bigmart-product-sales-factors). This project will focus on certain statistical features to gain valuable insights

# Objectives
1. Probability Distribution and Expected Values
2. Sales and Price
3. Correlation
4. Estimated Probabilities and Sales
5. Hypothesis Testing

# Flow
1. After immporting some libraries, this project goes on with some basic data cleansing, namely imputing outliers, imputing null and dropping duplicates (using a Class called Cleaning)
2. Each objective is mainly worked through two views, one a general view of all data and two a specific view of data with certain filter (e.g. Outlet_Type = 1)
3. This project uses an encoding process to convert categorical columns into numeric to allow some furhter statistical works
4. Adding a `Sales_Amount` since the dataset does not provide it

# Probability Distribution and Expected Values
Generating a list of expected values for all items using `Outlet_Location_Type` and `Outlet_Type` as indicators

# Sales and Price
Utilizing `jointplot` function to plot `Sales` and `Price` to gain insight into their density distribution

# Correlation
Plotting `Price` against `Sales` to understand correlation between those two and further `Price` against item characteristics to understand how they relate to `Price`

# Estimated Probabilities and Sales
Creating two functions to generate probabilities and sales using given sales and probabilities

# Hypothesis Testing
Devising hypothesis tests
