import matplotlib.pyplot as plt  # matplotlib.pyplot plots data
import pandas as pd  # pandas is a dataframe library

# ADDING & CHECKING CORRELATION
df = pd.read_csv("./data/pima-data.csv")


# print(df)

def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


plot_corr(df)
plt.show()

# DATA CLEANING
del df['skin']

diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# CHECKING DATA DISTRIBUTION

num_true: int = len(df.loc[df['diabetes']==True])
num_false = len(df.loc[df['diabetes']==False])
print("Number of true cases : {0} ({1:2.2f}%)".format(num_true, ((num_true / (num_true + num_false)) * 100)))
print("Number of false cases : {0} ({1:2.2f}%)".format(num_false, ((num_false / (num_true + num_false)) * 100)))

# DATA SPLITING
from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values  # predictor feature columns
Y = df[predicted_class_names].values  # predicted class {true : 1,false:0}
split_test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(X_train) / len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test) / len(df.index)) * 100))

#### Verifying predicted value was split correctly
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes']==1]),
                                                (len(df.loc[df['diabetes']==1]) / len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes']==0]),
                                                (len(df.loc[df['diabetes']==0]) / len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:]==1]),
                                                (len(Y_train[Y_train[:]==1]) / len(Y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:]==0]),
                                                (len(Y_train[Y_train[:]==0]) / len(Y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:]==1]),
                                                (len(Y_test[Y_test[:]==1]) / len(Y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:]==0]),
                                                (len(Y_test[Y_test[:]==0]) / len(Y_test) * 100.0)))

# POST SPLIT DATA PREPRATION

print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc']==0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp']==0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness']==0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin']==0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi']==0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred']==0])))
print("# rows missing age: {0}".format(len(df.loc[df['age']==0])))

#### Imputing with mean

from sklearn.impute import SimpleImputer

# Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")  # , axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# TRAINING INITIAL ALGORITHM - Naive Bayes

from sklearn.naive_bayes import GaussianNB

##Create object & train it with GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, Y_train.ravel())
