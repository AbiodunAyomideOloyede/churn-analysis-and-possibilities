import pandas as pd
df = pd.read_csv("C:/Users/MARCELA/OneDrive/Desktop/Python_Data(project)/big_messy_business_dataset.csv")

# df["Department"] = df["Department"].str.strip().str.capitalize()
# df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# df["Salary"] = pd.to_numeric(df["Salary"].replace({"80k": "80000", "N/A": None}), errors="coerce")
# df["JoinDate"] = pd.to_datetime(df["JoinDate"], errors="coerce")

# Data Cleaning Starts from here!
df["CustomerName"] = df["CustomerName"].str.strip().str.title() # Data Cleaning of "CustomerName" column to title case (Words with lower cases and some uppercases)
df["Age"] = pd.to_numeric(df["Age"].replace({"N/A": None}), errors="coerce").replace(["None", "nan", ""], pd.NA) # Data Cleaning of "Age" column to numbers 
df["Country"] = df["Country"].str.strip().str.title() # Data Cleaning of "Country" column to title case (Words with lower cases and some uppercases)
df["SignupDate"] = pd.to_datetime(df["SignupDate"], errors="coerce") # Data Cleaning of "SignupDate" column to Dates with consistent seperations
df["AnnualIncome"] = pd.to_numeric(df["AnnualIncome"].replace({"N/A": "NaN", "75k": "75000"}, regex=True), errors="coerce") # Data Cleaning of "AnnualIncome" column to numbers 
df["Gender"] = df["Gender"].replace({"N/A": None, "F": "Female", "M":"Male"}).str.strip().str.title() # Data Cleaning of "Gender" column to numbers to Male and Female (a categorical Data)
df["Churn"] = (df["Churn"].replace({"Yes": 1, "No": 0}).astype("float")) # Data Cleaning of "Churn" column to binary values (1 and 0)

# for Category_*
category_cols = [col for col in df.columns if col.startswith("Category_")]
for col in category_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"None": None})
    ) 
# Data Cleaning of Category_* columns to title case (Words with lower cases and some uppercases)

# For Metric_*
Metric_cols = [col for col in df.columns if col.startswith("Metric_")]
for col in Metric_cols:
    df[col] = pd.to_numeric(df[col].replace({"N/A": None}), errors="coerce") 
    # Data Cleaning of Metric_* columns to numbers 

# For Value_*
Value_cols = [col for col in df.columns if col.startswith("Value_")]
for col in Value_cols:
    df[col] = pd.to_numeric(df[col].replace({"N/A": None}), errors="coerce") 
    # Data Cleaning of Value_* columns to numbers

# # # Removing Missing Data 
threshold = 0.7  # Set threshold for missing data (50%)     
df = df.loc[:, df.isna().mean() < threshold]
for col in df.select_dtypes(include="number"):
     df[col].fillna(df[col].median(), inplace=True) 
#      # Filling missing numerical data with median values

for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode(), inplace=True)
# Filling missing categorical data with mode values
# Data Cleaning  ends here 

print(df.head(50)) 

# Data Analysis starts here 
print(df["Churn"].value_counts()) #This calculates the amount of customers that left 
print(df["Churn"].groupby(df["Gender"]).value_counts()) #This calculates the amount of customers that left base on gender
print(df["Churn"].groupby(df["AnnualIncome"]).value_counts()) #This calculates the amount of customers that left base on Annual Income 
print(df["Churn"].groupby(df["Age"]).value_counts()) #This calculates the amount of customers that left base on Age 
print((df["Age"]).mode()) #This calculates the plentiest Age group
print(df["Churn"].groupby(df["Country"]).value_counts()) #This calculates the amount of customers that left base on Country 
print(df["Churn"].groupby(df["AnnualIncome"]).value_counts(normalize=True)*100) #   This calculates the percentage of customers that left base on Annual Income
print(df["AnnualIncome"].groupby(df["Country"]).mean().sort_values(ascending=False)) # This calculates the average Annual Income base on Country
print(df["AnnualIncome"].groupby(df["Category_1"]).mean().sort_values(ascending=False)) # This calculates the average Annual Income base on Category_1
# file = df.to_csv("C:/Users/MARCELA/Downloads/cleaned_big_messy_business_dataset.csv")

# # Data Modelling Starts here! Model that calcaltes churn possibility
y = df["Churn"] # I am predicting Churn here, this is the selection of the column I am predicting
X = df[["Age", "AnnualIncome", "Gender", "Country", "Category_1"]] # I am predicting using this features   
X = pd.get_dummies(X, drop_first=True) # Converting categorical data to numerical data using One-Hot Encoding
df["CustomerTenure"] = (
    pd.Timestamp("today") - df["SignupDate"]
).dt.days 

metric_cols = [col for col in df.columns if col.startswith("Metric_")]

df["Metric_Mean"] = df[metric_cols].mean(axis=1) 
df["Metric_Max"] = df[metric_cols].max(axis=1)

X = df[
    [
        "Age",
        "AnnualIncome",
        "CustomerTenure",
        "Metric_Mean",
        "Metric_Max",
        "Gender",
        "Country",
        "Category_1"
    ]
]

df["Churn"].value_counts(normalize=True)
X = pd.get_dummies(X, drop_first=True)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



from sklearn.model_selection import train_test_split # Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # 80% training and 20% testing data

from sklearn.linear_model import LogisticRegression # Using Logistic Regression model for prediction
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) # Making predictions on the test set

from sklearn.metrics import accuracy_score, classification_report # Evaluating the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred)) # Printing the accuracy of the model
print(classification_report(y_test, y_pred)) # Printing a detailed classification report
# Modeling ends here

# Additional Analysis (Uncomment to use)
# df["SignupYear"] = df["SignupDate"].dt.year
# print(df.groupby("SignupYear").size())

# print(df.info())
# print(df.head())
# print(df.isna().count().sort_values(ascending=False).head(70))
# (df.isna().sum().gt(0).sum())

# print(df)
# print(df.describe())

# df = df.to_csv("C:/Users/MARCELA/Desktop/Cleaned_Dataset.csv", index=False)
