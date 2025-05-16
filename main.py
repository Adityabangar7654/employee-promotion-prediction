# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Data Load करो
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 3: Missing Values Check करो
print("Missing values in train data:\n", train_df.isnull().sum())
print("\nMissing values in test data:\n", test_df.isnull().sum())

# Step 4: Missing Values Fill करो

# Numerical Columns
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.drop(['employee_id', 'is_promoted'])

for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_df[col] = test_df[col].fillna(train_df[col].mean())  # test में भी train का mean लगाओ

# Categorical Columns
cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']

for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])

# Step 5: Encoding Categorical Columns (Label Encoding)
le = LabelEncoder()
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Step 6: Prepare Features and Target

X_train = train_df.drop(columns=['employee_id', 'is_promoted'])
y_train = train_df['is_promoted']

X_test = test_df.drop(columns=['employee_id'])

# Step 7: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Model Evaluation on train data (optional)
y_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))

# Step 9: Predict on test data (optional)
test_predictions = model.predict(X_test)

print("Test Predictions (First 10):", test_predictions[:10])
import pickle

# ऊपर वाला मॉडल ट्रेनिंग वाला कोड चलाने के बाद

# Model save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
# Model load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

#
# Format: [department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won?, avg_training_score]

sample_input = np.array([[2, 1, 0, 1, 0, 3, 35, 4, 10, 1, 80]])

# Prediction करो
pred = loaded_model.predict(sample_input)[0]

if pred == 1:
    print("Employee will be promoted")
else:
    print("Employee will NOT be promoted")

