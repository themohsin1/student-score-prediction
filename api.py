from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize the FastAPI app
app = FastAPI(title="🎓 Student Exam Score Prediction API")

# Load and train the model
data = pd.read_csv("student_scores.csv")
X = data[['Hours']]
y = data['Scores']
model = LinearRegression()
model.fit(X, y)

# Input schema
class StudyHours(BaseModel):
    hours: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Score Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict_score(item: StudyHours):
    hours = np.array([[item.hours]])
    prediction = model.predict(hours)[0]
    return {"predicted_score": round(float(prediction), 2)}
