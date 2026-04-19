# World Cup Match Predictor

Predicts win / draw / loss for international football matches using historical FIFA data.

## Tech stack
- Python
- Pandas
- Scikit-learn (Random Forest)
- Streamlit

## How to run

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python train.py

3. Launch the app:
streamlit run app.py

## Model
Trained on 32,000+ international matches from 1990 onwards.
Accuracy: 46% (three-class prediction — home win, draw, away win).
Features: historical win rate for each team, World Cup flag, neutral venue flag.