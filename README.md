# FIFA Match Outcome Predictor (For Fun Only)

This project is a playful experiment using historical FIFA World Cup data and a deep neural network to predict match outcomes based on team names, city, attendance, and referee.

## 🚨 Disclaimer

This model **is not accurate** and **should not be used for serious predictions**. It's a hobby project meant for fun, educational exploration, or messing around with data science and neural networks.

**Do not use this for betting, financial decisions, or anything serious.**  
If you do, that's entirely on you.

## ⚙️ What it Does

- Cleans and encodes data from `FIFA_World_Cup_1558_23.csv` (it's up to you to find this dataset online)
- Builds a deep neural network using Keras
- Trains the model to guess match outcomes (win/draw/loss)
- Allows predictions from command-line input (`user.py`)

## 🧠 Model Notes

- The architecture is deliberately overkill and experimental.
- The output layer currently uses a softmax with a single unit, which is **mathematically incorrect** for multi-class classification.
- Model performance is not evaluated with real metrics. It’s not useful beyond entertainment or testing code.

## 📂 Files

- `fifa.py`: Preprocesses the dataset and trains the model.
- `user.py`: Takes user input and generates a "prediction".
- `fifaclean.csv`, `fifawinner.csv`: Intermediate data files.
- `namesfile.dat`, `numfile.dat`: Encoded string mappings.
- `mi_model.h5`: Saved model.
- `user.csv`: Example input (not needed if running interactively).

## 📦 Installation

Install dependencies:
```bash
pip install pandas numpy scikit-learn keras tensorflow


python fifa.py       # Train the model
python user.py       # Use the model

```

Example of user.csv file:

```bash
Year    City    Home.Team.Name  Away.Team.Name  Attendance  Referee
2014    151.0   8.0 36.0    68034   360.0
2014    151.0   8.0 36.0    68034   360.0

```
