import pandas as pd
df=pd.read_csv('results.csv')
print(df.shape)        # how many rows and columns?
print(df.head(10))     # first 10 matches
print(df.columns)      # what columns exist
print(df.dtypes)       # what type is each column
print(df.isnull().sum())
print("Earliest:",df["date"].min())
print("Latest:",df["date"].max())
print(df["home_team"].nunique(),"unique home teams")


