from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# import the performance metrics library
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # matplotlib.pyplot plots data
import pandas as pd  # pandas is a dataframe library

# 1.ADDING & CHECKING CORRELATION


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

# 2.DATA CLEANING
del df['skin']

diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# CHECKING DATA DISTRIBUTION

num_true: int = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Number of true cases : {0} ({1:2.2f}%)".format(
    num_true, ((num_true / (num_true + num_false)) * 100)))
print("Number of false cases : {0} ({1:2.2f}%)".format(
    num_false, ((num_false / (num_true + num_false)) * 100)))

# 3.1.DATA SPLITING

feature_col_names = ['num_preg', 'glucose_conc',
                     'diastolic_bp', 'thickness', 'insulin', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values  # predictor feature columns
Y = df[predicted_class_names].values  # predicted class {true : 1,false:0}
split_test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(X_train) / len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test) / len(df.index)) * 100))

# 3.2.Verifying predicted value was split correctly
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]),
                                                (len(df.loc[df['diabetes'] == 1]) / len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]),
                                                (len(df.loc[df['diabetes'] == 0]) / len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 1]),
                                                (len(Y_train[Y_train[:] == 1]) / len(Y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 0]),
                                                (len(Y_train[Y_train[:] == 0]) / len(Y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 1]),
                                                (len(Y_test[Y_test[:] == 1]) / len(Y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 0]),
                                                (len(Y_test[Y_test[:] == 0]) / len(Y_test) * 100.0)))

# 4.POST SPLIT DATA PREPRATION

print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(
    len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(
    len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(
    len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(
    len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

# 4.1.Imputing with mean


# 4.2.Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")  # , axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# 5.1.TRAINING INITIAL ALGORITHM - Naive Bayes


# 5.2.Create object & train it with GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, Y_train.ravel())


# 6.1.Performance On Training data


# predict values using the training data
nb_predict_train = nb_model.predict(X_train)


print("Accuracy: {0:.3f}".format(metrics.accuracy_score(
    Y_train, nb_predict_train)))        # Accuracy
print()


# 6.2. Performance On Testin data

nb_predict_test = nb_model.predict(X_test)


print("Accuracy: {0:.3f}".format(metrics.accuracy_score(
    Y_test, nb_predict_test)))      # training metrics

# 6.3.Metrics

print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(Y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test, nb_predict_test))


# 7.1.TRAINING INITIAL ALGORITHM - Logistic Regression

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(
        C=C_val, solver='liblinear', random_state=42)
    lr_model_loop.fit(X_train, Y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(
    best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")

#7.2.Logisitic regression with class_weight='balanced'

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", solver='liblinear', random_state=42)
    lr_model_loop.fit(X_train, Y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))


plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")


lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, solver='liblinear', random_state=42)
lr_model.fit(X_train, Y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# 7.3.training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))
print(metrics.confusion_matrix(Y_test, lr_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(Y_test, lr_predict_test))
print(metrics.recall_score(Y_test, lr_predict_test))


#8.1.LogisticRegressionCV

from sklearn.linear_model import LogisticRegressionCV

# lr_cv_model = LogisticRegressionCV(n_jobs=-1, solver='liblinear', random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced")  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model = LogisticRegressionCV(n_jobs=-1, solver='liblinear', random_state=42, Cs=3, cv=10, refit=True,
                                   class_weight="balanced")  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model.fit(X_train, Y_train.ravel())


### Predict on Test data

lr_cv_predict_test = lr_cv_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(Y_test, lr_cv_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(Y_test, lr_cv_predict_test))

