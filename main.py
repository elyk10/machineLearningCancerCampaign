import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def main():

    df = pd.read_csv("data.csv")
    print(len(df))
    print(df["area_mean"].max())
    print(len(df[df["diagnosis"] == "M"]))
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.dropna(inplace = True)
    

    df.drop(["id"], axis = 1, inplace = True)
    


    labelEncoder = LabelEncoder()

    df["diagnosis"] = labelEncoder.fit_transform(df["diagnosis"])

    scaler = StandardScaler()
    
    features = df.iloc[:,1:] 
    df.iloc[:,1:] = scaler.fit_transform(features)

    df.to_csv("data_edit.csv", index = False)

    sns.pairplot(df, vars = df.columns[1:6], hue = "diagnosis", diag_kind= "kde")
    #plt.suptitle("Pair Plot for select Features")
    #plt.show()

    plt.figure(figsize = (12, 8))
    sns.heatmap(df.corr(), cmap = "coolwarm", annot = False)
    plt.title("Correlation Heatmap")
    #plt.show()

    plt.figure(figsize = (20, 10))
    dfMelted = df.melt(id_vars = "diagnosis", var_name = "Feature", value_name = "Value")
    sns.boxplot(x = "Feature", y = "Value", hue = "diagnosis", data = dfMelted)
    plt.xticks(rotation = 90)
    plt.title("Box Plots of Features by Diagnosis")
    #plt.show()

    correlations = df.corr()["diagnosis"].sort_values()

    importantFeatures = correlations[abs(correlations) > 0.6].index.tolist()
    print(importantFeatures)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[importantFeatures].corr(), annot = True, cmap= "coolwarm")
    plt.show()

    df2 = df[importantFeatures]
    y = df2["diagnosis"].values
    importantFeatures.remove("diagnosis")
    x = df2[importantFeatures].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size= 0.5, random_state= 0)

    print(len(x_train), len(x_val), len(x_test))

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    scores = []
    neighbours = range(1, 20)

    for i in neighbours:
        knn = KNeighborsClassifier(n_neighbors= i).fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))

    print(scores)

    knn = KNeighborsClassifier(n_neighbors= 5).fit(x_train, y_train)
    print("KNN SCORE: ", knn.score(x_test, y_test))

    rfc = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = "gini")
    rfc.fit(x_train, y_train)
    
    score = rfc.score(x_test, y_test)
    print("RFC SCORE: ", score)

    svc = SVC(random_state = 0, C = 0.2, gamma = "auto", kernel = "rbf")
    svc.fit(x_train, y_train)

    score = svc.score(x_test, y_test)
    print("SVC SCORE: ", score)



main()