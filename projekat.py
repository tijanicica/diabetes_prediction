import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



data = arff.loadarff('dataset_37_diabetes.arff')
diabetes_data = pd.DataFrame(data[0])
diabetes_data.head()
#diabetes_data.drop(columns=['class'], inplace=True)



diabetes_data['insu'] = diabetes_data['insu'].replace(0, float('nan'))
print(diabetes_data.head())

print("Null values: ", diabetes_data.isnull().sum())
diabetes_data = diabetes_data.dropna(subset=['insu'])
print("Provera")
print("Null values: ", diabetes_data.isnull().sum())
print(diabetes_data)
#diabetes_data.dtypes()

diabetes_data['class'] = diabetes_data['class'].map({b'tested_positive': 1.0, b'tested_negative': 0.0})

X = diabetes_data.drop(columns=['class'])
y = diabetes_data['class']

correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Coreelation map')
plt.show()

# Logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy logistic regression:", accuracy)
print("\nClassification Report logistic regression:")
print(classification_report(y_test, y_pred))

# Decision tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy decision tree:", accuracy)
print("\nClassification Report decision tree:")
print(classification_report(y_test, y_pred))

#Random forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy random forest:", accuracy)
print("\nClassification Report random forest:")
print(classification_report(y_test, y_pred))

#SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy svm model:", accuracy)
print("\nClassification Report svm model:")
print(classification_report(y_test, y_pred))


print("Max pedi", diabetes_data['pedi'].max())

print("Min pedi", diabetes_data['pedi'].min())

diabetes_data['preg_category'] = pd.cut(diabetes_data['preg'], bins=[-1, 0, 4, float('inf')], labels=['preg = 0', 'preg between 1 and 4', 'preg > 4'])
positive_cases = diabetes_data[diabetes_data['class'] == 1.0]
negative_cases = diabetes_data[diabetes_data['class'] == 0.0]

positive_counts = positive_cases['preg_category'].value_counts().reset_index()
positive_counts.columns = ['preg_category', 'positive_count']

plt.figure(figsize=(8, 6))
plt.pie(positive_counts['positive_count'], labels=positive_counts['preg_category'], autopct='%1.1f%%', colors=sns.color_palette("Spectral"))
plt.title('Number of Tested Positive Diabetes Cases by Pregnancy categories')
plt.show()

pregnancy_positive_count = positive_cases['preg'].value_counts().reset_index()
pregnancy_positive_count.columns = ['preg', 'positive_count']

# Plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(x='preg', y='positive_count', data=pregnancy_positive_count, marker='o', color='orange')
plt.xlabel('Number of Pregnancies')
plt.ylabel('Number of Tested Positive')
plt.title('Number of Tested Positive Diabetes Cases by Number of Pregnancies')
plt.grid(True)
plt.tight_layout()
plt.show()

#Devided age into categories
diabetes_data['age_category'] = pd.cut(diabetes_data['age'], bins=[20, 30, 40, float('inf')], labels=['21-30', '30-40', '>40'])

positive_cases = diabetes_data[diabetes_data['class'] == 1.0]

positive_counts_age = positive_cases['age_category'].value_counts().reset_index()
positive_counts_age.columns = ['age_category', 'positive_count_age']

plt.figure(figsize=(8, 6))
sns.barplot(x='age_category', y='positive_count_age', data=positive_counts_age, palette='Spectral')
plt.xlabel('Age Category')
plt.ylabel('Number of Tested Positive')
plt.title('Number of Tested Positive Diabetes Cases by Age Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

positive_counts_combined = positive_cases.groupby(['age_category', 'preg_category']).size().reset_index(name='positive_count')

positive_counts_pivot = positive_counts_combined.pivot(index='age_category', columns='preg_category', values='positive_count')

# Plot the results
plt.figure(figsize=(10, 6))
sns.heatmap(positive_counts_pivot, annot=True, cmap='Spectral', fmt='g',  color=sns.color_palette("Spectral"))
plt.xlabel('Pregnancy Category')
plt.ylabel('Age Category')
plt.title('Risk of Diabetes by Age and Pregnancy Category')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))

# Plot positive
plt.scatter(range(len(positive_cases)), positive_cases['pres'], color='orange', label='Tested Positive', alpha=0.7)

# Plot negative
plt.scatter(range(len(negative_cases)), negative_cases['pres'], color='pink', label='Tested Negative', alpha=0.7)

plt.xlabel('Patients')
plt.ylabel('Blood Pressure')
plt.title('Blood Pressure of Patients with Test Status')

plt.legend()
plt.xlim(0, 250)

plt.tight_layout()
plt.show()

positive_less_than_80 = positive_cases[(positive_cases['pres'] < 80) & (positive_cases['pres'] >= 0)]['pres'].count()
positive_between_80_100 = positive_cases[(positive_cases['pres'] >= 80) & (positive_cases['pres'] <= 100)]['pres'].count()
positive_above_100 = positive_cases[positive_cases['pres'] > 100]['pres'].count()

negative_less_than_80 = negative_cases[(negative_cases['pres'] < 80) & (negative_cases['pres'] >= 0)]['pres'].count()
negative_between_80_100 = negative_cases[(negative_cases['pres'] >= 80) & (negative_cases['pres'] <= 100)]['pres'].count()
negative_above_100 = negative_cases[negative_cases['pres'] > 100]['pres'].count()

print("Positive cases:")
print("  - Blood pressure less than 80:", positive_less_than_80)
print("  - Blood pressure between 80 and 100:", positive_between_80_100)
print("  - Blood pressure above 100:", positive_above_100)

print("\nNegative cases:")
print("  - Blood pressure less than 80:", negative_less_than_80)
print("  - Blood pressure between 80 and 100:", negative_between_80_100)
print("  - Blood pressure above 100:", negative_above_100)

#---

bins = [0, 200, float('inf')]
labels = ['Normal', 'High']

diabetes_data['insu_level'] = pd.cut(diabetes_data['insu'], bins=bins, labels=labels)

positive_cases = diabetes_data[diabetes_data['class'] == 1.0]
negative_cases = diabetes_data[diabetes_data['class'] == 0.0]

grouped_data_positive = positive_cases.groupby('insu_level').size()
grouped_data_negative = negative_cases.groupby('insu_level').size()

fig, axes = plt.subplots(1, 2, figsize=(16, 12))

axes[0].pie(grouped_data_positive, labels=grouped_data_positive.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
axes[0].set_title('Positive Diabetes Tests by Insulin Levels')

axes[1].pie(grouped_data_negative, labels=grouped_data_negative.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
axes[1].set_title('Negative Diabetes Tests by Insulin Levels')

plt.tight_layout()
plt.show()

bins = [0, 100, 150, float('inf')]

positive_cases.loc[:, 'glucose_range'] = pd.cut(positive_cases['plas'], bins=bins)
negative_cases.loc[:, 'glucose_range'] = pd.cut(negative_cases['plas'], bins=bins)


positive_counts = positive_cases['glucose_range'].value_counts(normalize=True)
negative_counts = negative_cases['glucose_range'].value_counts(normalize=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(positive_counts, labels=positive_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
axes[0].set_title('Positive cases')

axes[1].pie(negative_counts, labels=negative_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
axes[1].set_title('Negative cases')

plt.tight_layout()
plt.figure(figsize=(10, 6))

plt.scatter(positive_cases.index, positive_cases['plas'], c='orange', label='Tested Positive', alpha=0.7 )

plt.scatter(negative_cases.index, negative_cases['plas'], c='pink', label='Tested Negative', alpha=0.7)

positive_mean_plas = positive_cases['plas'].mean()
negative_mean_plas = negative_cases['plas'].mean()
difference = positive_mean_plas - negative_mean_plas

plt.text(0.5, 0.95, f'Difference: {difference:.2f}', color='black', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

plt.xlabel('Patients')
plt.ylabel('Glucose Level')
plt.title('Glucose Distribution in Patients with Positive and Negative Diabetes Test Results')
plt.legend()

plt.tight_layout()
plt.show()

positive_cases = diabetes_data[diabetes_data['class'] == 1.0]
negative_cases = diabetes_data[diabetes_data['class'] == 0.0]

plt.figure(figsize=(10, 6))
plt.hist(positive_cases['mass'], bins=20, color='red', alpha=0.5, label='Tested Positive', density=True)
plt.hist(negative_cases['mass'], bins=20, color='blue', alpha=0.5, label='Tested Negative', density=True)
plt.xlabel('BMI')
plt.ylabel('Density')
plt.title('BMI Distribution by Diabetes Test Result')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

data = arff.loadarff('dataset_37_diabetes.arff')
diabetes_data = pd.DataFrame(data[0])

diabetes_data['insu'] = diabetes_data['insu'].replace(0, float('nan'))

diabetes_data = diabetes_data.dropna(subset=['insu'])

diabetes_data['class'] = diabetes_data['class'].map({b'tested_positive': 1.0, b'tested_negative': 0.0})

bins = [0, 200, float('inf')]
labels = ['Normal', 'High']

diabetes_data['insu_level'] = pd.cut(diabetes_data['insu'], bins=bins, labels=labels)

positive_cases = diabetes_data[diabetes_data['class'] == 1.0]

high_level_counts = {
    'Glucose': positive_cases[positive_cases['plas'] > 150]['plas'].count(),
    'Blood Pressure': positive_cases[positive_cases['pres'] > 100]['pres'].count(),
    'BMI': positive_cases[positive_cases['mass'] > 30]['mass'].count(),
    'Insulin': positive_cases[positive_cases['insu'] > 200]['insu'].count()
}

counts_df = pd.DataFrame.from_dict(high_level_counts, orient='index', columns=['Count'])

plt.figure(figsize=(10, 6))
counts_df.plot(kind='bar', color='skyblue', legend=None)
plt.xlabel('Factor')
plt.ylabel('Count')
plt.title('Number of Positive Diabetes Tests with High Factor Levels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

positive_counts_pedi = diabetes_data.groupby('pedi')['class'].count().reset_index()

positive_counts_pedi_positive = positive_counts_pedi[positive_counts_pedi['class'] == 1.0]

plt.figure(figsize=(10, 6))
sns.histplot(data=positive_counts_pedi_positive, x='pedi', bins=20, color='skyblue', edgecolor='skyblue')
plt.xlabel('Pedi Value')
plt.ylabel('Count')
plt.title('Distribution of Tested Positive Diabetes Cases by Pedi Value')
plt.grid(True)
plt.tight_layout()
plt.show()

def count_positive_cases(data, column_name):
    positive_cases = data[data['class'] == 1.0]
    positive_counts = positive_cases.groupby(column_name).size().reset_index(name='positive_count')
    positive_counts_dict = dict(zip(positive_counts[column_name], positive_counts['positive_count']))
    return positive_counts_dict

mass_positive_counts = count_positive_cases(diabetes_data, 'mass')
skin_positive_counts = count_positive_cases(diabetes_data, 'skin')

mass_positive_counts_df = pd.DataFrame(list(mass_positive_counts.items()), columns=['mass', 'positive_count_mass'])
skin_positive_counts_df = pd.DataFrame(list(skin_positive_counts.items()), columns=['skin', 'positive_count_skin'])

plt.figure(figsize=(10, 6))

plt.plot(mass_positive_counts_df['mass'], mass_positive_counts_df['positive_count_mass'], label='Positive Cases by Mass', color='orange')

plt.plot(skin_positive_counts_df['skin'], skin_positive_counts_df['positive_count_skin'], label='Positive Cases by Skin',   color='skyblue')

plt.xlabel('Mass and Skin')
plt.ylabel('Number of Positive Cases')
plt.title('Positive Diabetes Cases by Mass and Skin')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






