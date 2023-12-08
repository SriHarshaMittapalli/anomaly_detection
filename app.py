import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import plotly.graph_objects as go

st.markdown("<h1><a href='https://research.unsw.edu.au/projects/unsw-nb15-dataset' style='text-decoration: none; color: black;'>UNSW_NB15 Dataset</a></h1>", unsafe_allow_html=True)

st.write("The UNSW-NB15 dataset combines synthetic and real-world network data. This dataset has nine types of attacks namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms.")

# Reading datasets
dfs = []
dfs.append(pd.read_csv('UNSW_NB15_training-set.csv'))
dfs.append(pd.read_csv('UNSW_NB15_testing-set.csv'))
df = pd.concat(dfs).reset_index(drop=True)  # Concat all to a single df

st.write("Printing the first 10 rows in the dataframe.")
st.dataframe(df.head())

# st.write("Checking whether there are null values in the dataframe.")
# st.write(df.isnull().sum())

st.write("We are removing unnecessary columns; specifically, we are dropping the 'id' and 'attack_cat' columns.")

st.write("As we can see below, in the UNSW-NB15 Dataset, there are 9 attacks")
st.write(df["attack_cat"].unique())

df.drop(['id','attack_cat'],axis=1,inplace=True) # dropping these columns from the dataframe

st.header("Data Pre-Processing")

st.write("In Data Pre-Processing, first we are checking if there are any null values, second we are encoding categorical columns")

df_numeric = df.select_dtypes(include=[np.number]) # Select numeric columns
df_cat = df.select_dtypes(exclude=[np.number])

# Clamp extreme values and apply log transformation
for feature in df_numeric.columns:

    if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10: df[feature] = np.where(df[feature] < df_numeric[feature].quantile(0.95), df[feature], df_numeric[feature].quantile(0.95))

    if df_numeric[feature].nunique() > 50:
        if df_numeric[feature].min() == 0: df[feature] = np.log(df[feature] + 1)
        else: df[feature] = np.log(df[feature])

for feature in df_cat.columns:
    if df_cat[feature].nunique() > 6: df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

# Feature Selection
best_features = SelectKBest(score_func = chi2, k = 'all')

X = df.iloc[:,4:-2]
y = df.iloc[:,-1] # Indicates that the last column is considered the target variable

st.header("Feature Selection")

fit = best_features.fit(X,y)

feature_scores = pd.DataFrame({'feature': X.columns, 'score': fit.scores_})
feature_scores.sort_values(by='score', ascending=True, inplace=True)

fig = go.Figure(go.Bar(x=feature_scores['score'].head(20), y=feature_scores['feature'].head(20), orientation='h')) # Creating the Plotly bar chart
fig.update_layout(title="Top 20 Features by Score", height=800, showlegend=False) # Update the layout

# Display the chart using Streamlit
st.plotly_chart(fig)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X.head()
feature_names = list(X.columns)
np.shape(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

np.shape(X)

for label in list(df_cat['state'].value_counts().index)[::-1][1:]: feature_names.insert(0,label)
for label in list(df_cat['service'].value_counts().index)[::-1][1:]: feature_names.insert(0,label)
for label in list(df_cat['proto'].value_counts().index)[::-1][1:]: feature_names.insert(0,label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

# 6 + 5 + 6 unique = 17, therefore the first 17 rows will be the categories that have been encoded, start scaling from row 18 only.
sc = StandardScaler()
X_train[:, 18:] = sc.fit_transform(X_train[:, 18:])
X_test[:, 18:] = sc.transform(X_test[:, 18:])

model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','time to train','time to predict','total time'])

def evaluate_model(model_type, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    y_predictions = model.predict(X_test)
    end_predict = time.time()
    return {
        'Model': model_type, 'Accuracy': accuracy_score(y_test, y_predictions), 'Recall': recall_score(y_test, y_predictions, average='weighted'), 'Precision': precision_score(y_test, y_predictions, average='weighted'), 'F1 Score': f1_score(y_test, y_predictions, average='weighted'), 'Training Time': end_train - start, 'Prediction Time': end_predict - end_train, 'Total Time': end_predict - start
    }

def evaluate_all_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(), 'kNN': KNeighborsClassifier(n_neighbors=3), 'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap=True), 'Gradient Boosting Classifier': GradientBoostingClassifier()
    }
    model_performance = [evaluate_model(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]
    return model_performance

# Call the function with your X_train, X_test, y_train, y_test data
st.dataframe(evaluate_all_models(X_train, X_test, y_train, y_test))