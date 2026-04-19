import streamlit as st
import pickle
import numpy as np

with open("model.pkl","rb")as f:
    model=pickle.load(f)
with open("team_form.pkl","rb")as f:
    team_form=pickle.load(f)

all_teams= sorted(team_form.keys())

st.set_page_config(page_title=" International Football Match Predictor", page_icon="⚽")
st.title("International Football Match Predictor")
st.markdown("Select 2 teams and see who the model thinks will win")
col1, col2=st.columns(2)

with col1:
    st.subheader("Home Team")
    home_team= st.selectbox("Select home team",all_teams,index=all_teams.index("Brazil"))
with col2:
    st.subheader("Away Team")
    away_team=st.selectbox("Select Away Team",all_teams,index=all_teams.index("Argentina"))

st.divider()
is_worldcup=st.toggle("FIFA WORLD CUP match?",value=True)
neutral=st.toggle("Neutral Venue?",value=True)

st.divider()
if st.button("Predict Result",type="primary"):
    if home_team==away_team:
        st.warning("Please select two different Teams")
    else:
        home_form=team_form.get(home_team,0.5)
        away_form=team_form.get(away_team,0.5)
        
        features=np.array([[
            int(is_worldcup),
            int(neutral),
            home_form,
            away_form

        ]
        ])
        proba = model.predict_proba(features)[0]
        classes = model.classes_
        st.write(classes)

        prob_map=dict(zip(classes,proba))

        home_win_pct= prob_map["home_win"]*100
        draw_pct= prob_map["draw"]*100
        away_win_pct=prob_map["away_win"]*100

        st.subheader("Prediction")
        r1,r2,r3=st.columns(3)
        r1.metric(f"{home_team} Win",f"{home_win_pct:.1f}%")
        r2.metric("Draw",f"{draw_pct:.1f}%")
        r3.metric(f"{away_team} Win",f"{away_win_pct}%")

        st.markdown(f"**{home_team}** win")
        st.progress(home_win_pct/100)
        st.markdown("**Draw**")
        st.progress(draw_pct/100)
        st.markdown(f"**{away_team}** win")
        st.progress(away_win_pct/100)

        predicted = max(prob_map, key=prob_map.get)
        label_map = {
            "home_win": f"🏠 {home_team} win",
            "draw":      "🤝 Draw",
            "away_win": f"✈️ {away_team} win"
        }
        st.success(f"Model predicts: **{label_map[predicted]}**")

        with st.expander("Why this Prediction?"):
            st.write(f"**{home_team} historical win rate:**{home_form:.2f}")
            st.write(f"**{away_team} historical win rate:**{away_form:.2f}")
            st.write(f"**World Cup Match:**{is_worldcup}")
            st.write(f"**Neutral venue:**{neutral}")
            st.write("The model relies almost entirely on historical win rates — "
                     "the team that wins more often is more likely to win here too.")
            