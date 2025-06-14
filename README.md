Customer Churn Prediction System
A machine learning project designed to predict whether a customer will churn (i.e., leave the service) based on features such as contract type, charges, tenure, and more. The system uses multiple models, advanced preprocessing techniques, and visualizations to provide actionable business insights.
Customer_Churn_Prediction/
│
├── customer_churn.csv              # Dataset
├── churn_prediction.py             # Python source code
├── README.md                       # Project explanation
├── shap_summary.png                # SHAP feature importance plot
├── correlation_heatmap.png        # Feature correlation heatmap
├── churn_distribution.png         # Target class distribution
├── monthly_charges_by_churn.png   # Boxplot by churn status
├── roc_curve_comparison.png       # ROC curve of models
└── demo.mp4                        # (Optional) Video demo

🎯 Objective
To build a predictive machine learning system that classifies customers based on their likelihood to churn, and provides visual and statistical insights for reducing churn.

🧠 Key Features
Preprocessing of real-world telecom customer data

Class imbalance handled using SMOTE

Feature engineering with new interaction terms

Model training using:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

XGBoost

Voting Classifier (Ensemble)

Evaluation using Accuracy, Precision, Recall, F1 Score, ROC-AUC

Visualizations for data analysis and model evaluation

SHAP interpretability for explaining model predictions

📦 Dataset
Source: Kaggle or UCI repository

Attributes: customerID, gender, SeniorCitizen, Partner, tenure, Contract, MonthlyCharges, TotalCharges, Churn, and more.

🛠 Technologies Used
Python

Pandas, NumPy

Scikit-learn

XGBoost

Seaborn, Matplotlib

SHAP (for explainability)

SMOTE (for imbalance handling)

🚀 Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Python Script

bash
Copy
Edit
python churn_prediction.py
📊 Visualizations
correlation_heatmap.png – Shows inter-feature relationships

churn_distribution.png – Shows class imbalance

monthly_charges_by_churn.png – Highlights billing vs churn behavior

roc_curve_comparison.png – Comparison of model ROC curves

shap_summary.png – Key features affecting churn

✅ Model Evaluation Summary
Model	Accuracy	Precision	Recall	F1 Score	AUC
Logistic Regression	~0.79	~0.72	~0.65	~0.68	~0.81
Random Forest	~0.82	~0.76	~0.72	~0.74	~0.86
SVM	~0.80	~0.74	~0.68	~0.71	~0.84
XGBoost	~0.84	~0.78	~0.75	~0.76	~0.89
Ensemble (Best)	~0.86	~0.81	~0.77	~0.79	~0.91

💡 Business Recommendations
Incentivize long-term contracts to reduce month-to-month churners.

Focus on retention during the first 6 months of customer tenure.

Monitor high-bill customers and offer personalized plans.

Improve customer service for users with frequent support calls.

Integrate churn prediction into your CRM for real-time alerts.
