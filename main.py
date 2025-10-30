import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def main():

    df = pd.read_csv("data.csv")

    
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
    plt.show()

    plt.figure(figsize = (20, 10))
    dfMelted = df.melt(id_vars = "diagnosis", var_name = "Feature", value_name = "Value")
    sns.boxplot(x = "Feature", y = "Value", hue = "diagnosis", data = dfMelted)
    plt.xticks(rotation = 90)
    plt.title("Box Plots of Features by Diagnosis")
    plt.show()

main()