import pandas as pd
import numpy as np
df=pd.read_csv('results.csv')
df = df.dropna(subset=["home_score", "away_score"])
df["home_score"]=df["home_score"].astype(int)
df["away_score"]=df["away_score"].astype(int)
df["date"]=pd.to_datetime(df["date"])
df=df[df['date'].dt.year>=1990].copy()
df['result']=np.where(df['home_score']>df['away_score'],"home_win",np.where(df["home_score"]==df['away_score'],"draw","away_win"))
df['is_worldcup']=df['tournament'].str.contains('FIFA World Cup',na=False).astype(int)
df['neutral']=df['neutral'].astype(int)
df= df.sort_values("date").reset_index(drop=True)
def rolling_win_rate(df,team_col,result_val,new_col):
    win_rates={}
    rates=[]
    for _,row in df.iterrows():
        team= row[team_col]
        result=row["result"]
        rate=win_rates.get(team,0.5)
        rates.append(rate)
        if team not in win_rates:
            win_rates[team]=0.5
        win_rates[team]=0.9*win_rates[team]+0.1*(1 if result==result_val else 0)
    df[new_col]= rates
    return df
df= rolling_win_rate(df,"home_team","home_win","home_team_form")
df =rolling_win_rate(df,"away_team","away_win","away_team_form")

print("Shape after cleaning:")
print(df.shape)
print("\n Target Distribution:")
print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True),round(2))
print('\n Features Preview')
print((df[["date", "home_team", "away_team", "result",
          "is_worldcup", "neutral",
          "home_team_form", "away_team_form"]].head(10)))
print("\n world cup matches")
wc=df[df['is_worldcup']==1]
print(f"{len(wc)}World cup matches in the dataset")
print(wc["result"].value_counts())
