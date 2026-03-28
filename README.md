# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select features and target variable.
2.Split the data into training and testing sets and apply feature scaling.
3.Train the SVM model using GridSearchCV to find the best parameters.
4.Predict the test data and evaluate the model using accuracy and classification report.
5.Generate and display the confusion matrix to visualize model performance.
## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: V MUKESHKUMAR
RegisterNumber:25012063

import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'
X = data[features]
y = data[target]
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Name: MUKESHKUMAR.V ")
print("Register Number: 25012063 ")
print(" Best Parameters:",grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Name: MUKESHKUMAR.V ")
print("Register Number: 25012063 ")
print("Accuracy:",accuracy)
print(" Classification Report:\n", classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

*/
```

## Output:
<img width="855" height="684" alt="Screenshot 2026-03-17 200541" src="https://github.com/user-attachments/assets/42ca952b-ddc2-46e0-95e2-c89ca958c9c8" />
<img width="623" height="71" alt="Screenshot 2026-03-17 200551" src="https://github.com/user-attachments/assets/c35ef3ef-5772-4863-936a-1dadd1dd626c" />
<img width="567" height="269" alt="Screenshot 2026-03-17 200559" src="https://github.com/user-attachments/assets/f76f0a6c-6f82-4107-9ac7-1555e056a9ab" />
<img width="807" height="581" alt="Screenshot 2026-03-17 200610" src="https://github.com/user-attachments/assets/513808f1-88bb-4d0e-be96-edde6e1a12e6" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
