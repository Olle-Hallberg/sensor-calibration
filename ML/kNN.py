import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.preprocessing as skl_pre
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_met

###### data ######
df = pd.read_csv('training_data_VT2026.csv')  # create dataframe

X = df.drop(columns=['increase_stock']) # features
y = df['increase_stock']                # demand


### tuning ###
# best k through 10-fold cross-validation
def n_fold_cross_val(X, y, n_folds, num_k, random_state):
  cv = skl_ms.StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)  # train/validation indices
  K = np.arange(1, num_k+1)                                                               # k values
  mis = np.zeros(len(K))                                                                  # misclassification

  for train_index, val_index in cv.split(X, y):
    # acquire test and validation data
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # scale data
    scaler = skl_pre.StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # test all k for current fold
    for j, k in enumerate(K):
      model = skl_nb.KNeighborsClassifier(n_neighbors=k)
      model.fit(X_train_scaled, y_train)
      prediction = model.predict(X_val_scaled)
      mis[j] += np.mean(prediction != y_val)  # store misclassification for current fold and k

  mis /= n_folds # devide by 10 scince mis incliudes sum of misclassification from 10 folds
  best_k = K[np.argmin(mis)]

  return(best_k, mis)

# plot and print tuning result
best_k, mis = n_fold_cross_val(X, y, n_folds=10, num_k=200, random_state=42)
print(f'Best k: {best_k}')

K = np.arange(1, 201)
plt.plot(K, mis)
plt.xlabel('Number of neighbors k')
plt.xticks(fontsize=11)
plt.ylabel('Validation error')
plt.yticks(fontsize=11)
# plt.show()  # uncomment to plot


###### evaluation ######
# scaled train and test data
np.random.seed(42)
trainI = np.random.choice(1600, 1280, replace=False)
trainIndex = df.index.isin(trainI)
train = df.iloc[trainIndex] # training set (80%)
test = df.iloc[~trainIndex] # test set (20%)

X_train = train.drop(columns=['increase_stock'])
y_train = train["increase_stock"]
X_test = test.drop(columns=["increase_stock"])
y_test = test["increase_stock"]

scaler = skl_pre.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# implement model
model = skl_nb.KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
prediction = model.predict(X_test_scaled)

# confusion matrix
print("Confusion Matrix:\n")
print(pd.crosstab(prediction, y_test), "\n")

CM = skl_met.confusion_matrix(y_test, prediction, labels=model.classes_)
disp = skl_met.ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.tight_layout()
# plt.show()  # uncomment to plot

# accuracy
print(f"Accuracy: {np.mean(prediction == y_test)}")
print("F1 macro:", skl_met.f1_score(y_test, prediction, average="macro"))