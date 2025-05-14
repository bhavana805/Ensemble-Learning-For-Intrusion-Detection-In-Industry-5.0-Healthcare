import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, 
                             RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data = pd.read_csv('healthcare_iomt_dataset.csv')

# Preprocessing
# Encode categorical variables
categorical_cols = ['IoMT_Type', 'Location', 'Auth_Method', 'Data_Format', 'OS_Type', 'Alert_Level']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target variable
le_attack = LabelEncoder()
data['Attack_Type'] = le_attack.fit_transform(data['Attack_Type'])

# Split data into features and target
X = data.drop(['Device_ID', 'Attack_Type'], axis=1)
y = data['Attack_Type']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize individual models
extra_trees = ExtraTreesClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Create a dictionary of models
models = {
    "Extra Trees": extra_trees,
    "Gradient Boosting": gradient_boosting,
    "Random Forest": random_forest,
    "Logistic Regression": logistic_regression,
    "MLP": mlp
}

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('et', extra_trees),
        ('gb', gradient_boosting),
        ('rf', random_forest),
        ('lr', logistic_regression),
        ('mlp', mlp)
    ],
    voting='soft'  # Use 'soft' for probability voting
)

# Add Voting Classifier to models dictionary
models["Voting Classifier"] = voting_clf

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    
    # Save the trained model
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Set style
sns.set(style="whitegrid")

# Melt the dataframe for easier plotting
melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Create bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df)
plt.title('Model Performance Comparison (Including Voting Classifier)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()

# Identify best model based on F1 score
best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
best_model = models[best_model_name]

# Save the best model and preprocessing objects
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_attack, 'attack_label_encoder.pkl')

print(f"Best model: {best_model_name}")

# Load the saved objects
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
le_attack = joblib.load('attack_label_encoder.pkl')

# Prepare input data
input_data = {
    'IoMT_Type': 'Implant',
    'Location': 'ICU',
    'Data_Transferred_MB': 187.58,
    'Access_Time(ms)': 4157,
    'Auth_Method': 'Two-Factor',
    'Requests_Per_Minute': 96,
    'Data_Format': 'HL7',
    'OS_Type': 'Android',
    'Alert_Level': 'Medium'
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input
for col in categorical_cols:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Scale the numerical features
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = best_model.predict(input_scaled)
predicted_attack = le_attack.inverse_transform(prediction)[0]

print(f"Predicted Attack Type: {predicted_attack}")