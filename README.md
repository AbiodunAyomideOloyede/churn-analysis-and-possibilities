# churn-analysis-and-possibilities
The project look at churn with different demographic reasons and the possibility that new customers will churn. 
The file contains 300 rows and 200 columns. The columns range from CustomerID to Metric_192.
The file contains too many inconsistencies that need to be cleaned. Each rows has data that are not correct owing to different reasons like wrong input, different system of input.
Cleaning started with column "CustomerName" All the Data therein were converted to title case.
In column "Age", all the data therein are converted to numeric to ensure that all data are consistent.
Column "Country", the Data therein were also converted to title case.
Column "SignupDate", The Data therein were converted to Datetime.
"AnnualIncome", Some inconsistencies like "75K" were replaced.
Column "Gender", The "M" and "F" were replace with "Male" and "Female" and all the lowere cases were converted to title case.
Column "Churn", The "Yes" and "No" there in are converted to numeric and to float. 
