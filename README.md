🚌 TTC Transit Delay Classification with Real-Time Data Insights
📘 Overview

TTC Transit Delay Classification is a machine learning project designed to predict whether a bus will arrive on time or be delayed, based on real-time operational and contextual data.
The project aims to improve transit reliability and provide actionable insights using advanced classification techniques.
After testing multiple models, the Random Forest Classifier delivered the most accurate and stable results.

🎯 Objective

To classify bus trips as On-Time or Delayed using features derived from schedule data, direction, route, and timing—helping in better operational decision-making and transit management.

🧠 Machine Learning Workflow
1. Data Preprocessing

Cleaned and standardized dataset

Handled missing values in route and direction columns

Combined date and time into DateTime

Extracted meaningful time features:

Hour, Minute, Month, Year

DayOfWeek, TimeOfDay, IsWeekend

Created additional derived columns:

AvgRouteDelay, AvgDirectionDelay

Dropped unnecessary columns like Vehicle and Date

2. Feature Engineering

Identified categorical and numerical features

Applied OneHotEncoding for categorical features

Scaled numeric features with StandardScaler

3. Model Building

Tested multiple algorithms:

Logistic Regression

Decision Tree

Random Forest ✅ (Best Model)

Gradient Boosting

XGBoost

LightGBM

CatBoost

4. Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Random Forest achieved the highest performance across all metrics.
Feature importance analysis revealed "Min Gap" as the most influential feature in determining delay likelihood.

5. Deployment

Trained model saved using joblib

Developed a Streamlit-based web app (app.py) for real-time prediction

Executed successfully in both Google Colab and Terminal environment

⚙️ Tech Stack
Category	Technologies
Language	Python
Libraries	pandas, numpy, scikit-learn, joblib, streamlit, matplotlib
Model	Random Forest Classifier
Environment	Google Colab, Local Terminal
Version Control	Git + GitHub
📂 Project Structure
TTC-Transit-Delay-Classification-with-Real-Time-Data-Insights/
│
├── data/
│   └── transit_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── app.py
├── bus_delay_rf_model.pkl
├── route_averages.json
├── direction_averages.json
├── requirements.txt
├── README.md
└── images/
    └── feature_importance.png

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/karthikpappala/TTC-Transit-Delay-Classification-with-Real-Time-Data-Insights.git
cd TTC-Transit-Delay-Classification-with-Real-Time-Data-Insights

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

📊 Results

Best Model: Random Forest

Accuracy: 99.6435
F1 Score: 99.8129

Most Important Feature: Min Gap

Insight: Buses with higher Min Gap values have greater delay probability.

🔮 Future Enhancements

Integration with real-time GPS and traffic data

Cloud deployment on Streamlit Cloud / AWS

Add interactive dashboards for route performance monitoring

Model retraining pipeline for adaptive learning
