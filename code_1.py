# Numpy
import numpy as np
#Pandas
import pandas as pd
#matplotlib
import matplotlib.pyplot as plt

# Load pima indians dataset
df = pd.read_csv("bank.csv")

print("head")
# Display first five records of data
print(df.head())

print("tail")
# Display last five records of the data
print(df.tail())

print("sample")
# Display randomly any number of records of data
print(df.sample(10))

print()
#List the types of all columns.
df.dtypes


#finding out if the dataset contains any null value
df.info()

# Statistical summary
df.describe()



#Number of rows and columns before dropping duplicates
df.shape

df=df.drop_duplicates()

#Number of rows and columns after dropping duplicates
df.shape

# Count of null, values
# check the missing values in any column
# Display number of null values in every column in dataset
df.isnull().sum()


df.describe()



# Check if the DataFrame and column exist
print("Outcome value counts:\n", df['Bankrupt?'].value_counts())

# Creating subplots
f, ax = plt.subplots(1, 2, figsize=(10, 5))

# Pie chart
df['Bankrupt?'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')

# Count plot using matplotlib
df['Bankrupt?'].value_counts().plot(kind='bar', ax=ax[1])
ax[1].set_title('Outcome')

# Print counts
N, P = df['Bankrupt?'].value_counts()
print('Not Bankrupt (0):', N)
print('Bankrupt (1):', P)

# Display the plot
plt.grid(True)  # Ensure grid lines are enabled
plt.show()


# Histogram of each feature
df.hist(bins=8,figsize=(8,8))
plt.show()


# Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize = (20, 20))
plt.show()


# Compute the correlation matrix
corrmat = df.corr()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Create the heatmap
cax = ax.imshow(corrmat, cmap='RdYlGn', interpolation='nearest')

# Add color bar
cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
cbar.set_label('Correlation Coefficient')

# Add annotations
for (i, j), val in np.ndenumerate(corrmat):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

# Set ticks and labels
ax.set_xticks(np.arange(len(corrmat.columns)))
ax.set_yticks(np.arange(len(corrmat.columns)))
ax.set_xticklabels(corrmat.columns, rotation=90)
ax.set_yticklabels(corrmat.columns)

# Set title and labels
plt.title('Correlation Heatmap')
plt.tight_layout()

# Display the plot
plt.show()


target_name = 'Bankrupt?'
# Separate object for target feature
y=df[target_name]
#Separate Object for Input Features
X = df.drop(target_name, axis=1)

# Apply Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
SSX = scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(SSX, y, test_size=0.2,random_state=7)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)

lr_pred=lr.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of Logistic Regression",accuracy_score(y_test,lr_pred)*100)


# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_lr = confusion_matrix(y_test, lr_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#Assuming you have the predicted probabilities for the positive class
lr_probs = lr.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, lr_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, lr_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Logistic Regression (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of KNN",accuracy_score(y_test,knn_pred)*100)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_knn = confusion_matrix(y_test, knn_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the predicted probabilities for the positive class
knn_probs = knn.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, knn_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, knn_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('KNN (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#NAIVE-BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_pred=nb.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of Naive Bayes",accuracy_score(y_test,nb_pred)*100)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_nb = confusion_matrix(y_test, nb_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the predicted probabilities for the positive class
nb_probs = nb.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, nb_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, nb_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Naive Bayes (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
sv = SVC()
sv.fit(X_train, y_train)

sv_pred=sv.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of SVM",accuracy_score(y_test,sv_pred)*100)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_sv = confusion_matrix(y_test, sv_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_sv, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming sv is your trained SVM model without probability=True
sv_probs = sv.decision_function(X_test)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, sv_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, sv_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('SVM (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_pred=dt.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of Decision Tree",accuracy_score(y_test,dt_pred)*100)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_dt = confusion_matrix(y_test, dt_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the predicted probabilities for the positive class
dt_probs = dt.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, dt_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, dt_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Decision Tree (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred=rf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of Random Forest",accuracy_score(y_test,rf_pred)*100)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_rf = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Bankrupt', 'Bankrupt'])
disp.plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming you have the predicted probabilities for the positive class
rf_probs = rf.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, rf_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Random Forest (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


