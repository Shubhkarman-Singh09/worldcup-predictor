import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

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
def rolling_win_rate(df, team_col, result_val, new_col):
    df = df.copy()           
    win_rates = {}
    rates = []
    for _, row in df.iterrows():
        team = row[team_col]
        result = row["result"]
        rate = win_rates.get(team, 0.5)
        rates.append(rate)
        win_rates[team] = 0.9 * win_rates.get(team, 0.5) + 0.1 * (1 if result == result_val else 0)
    df[new_col] = rates
    return df
df= rolling_win_rate(df,"home_team","home_win","home_team_form")
df =rolling_win_rate(df,"away_team","away_win","away_team_form")
#Model Training

features=["is_worldcup","neutral","home_team_form","away_team_form"]
X=df[features]
y=df["result"]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"Training on {len(X_train)}matches")
print(f"testing on {len(X_test)}matches")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print("Model trained.")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n Classification Report")
print(classification_report(y_test,y_pred))

print("\n Confusion Matrix")
cm= confusion_matrix(y_test,y_pred,labels=["home_win","draw","away_win"])
print(pd.DataFrame(cm,
                   index=["actual:home_win","actual:draw","actual: away_win"],
columns=["pred:home_win","pred:draw","pred:away_win"])
)

#Features
print("\n feature Importance")
for feat,imp in sorted(zip(features,clf.feature_importances_),key=lambda x:x[1],reverse=True):
    bar = "-" * int(imp * 40)
    print(f"{feat:<20} {bar}  {imp:.3f}")

#Saving the Model

with open("model.pkl","wb")as f:
    pickle.dump(clf,f)
team_form={}
for _,row in df.iterrows():
    team_form[row["home_team"]]=row["home_team_form"]
    team_form[row["away_team"]]=row["away_team_form"]

with open("team_form.pkl","wb") as f:
    pickle.dump(team_form,f)

print("\nSaved model.pkl and team_form.pkl")
print(f"Team form ratings saved for {len(team_form)} teams")
