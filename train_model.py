import pandas as pd
import joblib
from models import StudentRiskPredictor

# Load dataset
df = pd.read_csv("data.csv")

# Clean columns
df.columns = df.columns.str.strip().str.upper()

df = df[['CAT1','CAT2','QUIZ1','QUIZ2','QUIZ3','ATTENDANCE PERCENTAGE','FAT']]

df.rename(columns={
    'CAT1': 'mid1',
    'CAT2': 'mid2',
    'QUIZ1': 'quiz1',
    'QUIZ2': 'quiz2',
    'QUIZ3': 'quiz3',
    'ATTENDANCE PERCENTAGE': 'attendance',
    'FAT': 'final_exam'
}, inplace=True)

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Train model
model = StudentRiskPredictor()
model.train_models(df)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")