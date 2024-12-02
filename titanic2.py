import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousTitanicLogistics():
    # Step 1: Load data
    pd.set_option('display.max_rows', None)  #  rows
    pd.set_option('display.max_columns', None)  # columns

    titanic_data = pd.read_csv('MarvellousTitanicDataset1.csv')

    print("All entries from the loaded dataset:")
    print(titanic_data)

    print("Number of passengers: " + str(len(titanic_data)))

    # Step 2: Analyzing data
    print("Visualization: Survived and non-survived passengers")
    sns.countplot(data=titanic_data, x="Survived").set_title("Survived and Non-Survived Passengers")
    plt.show()

    print("Visualization: Survived and non-survived passengers based on gender")
    sns.countplot(data=titanic_data, x="Survived", hue="Sex").set_title("Survived and Non-Survived Passengers by Gender")
    plt.show()

    print("Visualization: Survived and non-survived passengers based on passenger class")
    sns.countplot(data=titanic_data, x="Survived", hue="Pclass").set_title("Survived and Non-Survived Passengers by Class")
    plt.show()

    print("Visualization: Age distribution of passengers")
    titanic_data["Age"].plot.hist().set_title("Age Distribution of Passengers")
    plt.show()

    print("Visualization: Fare distribution of passengers")
    titanic_data["Fare"].plot.hist().set_title("Fare Distribution of Passengers")
    plt.show()

    # Step 3: Data Cleaning
    if "zero" in titanic_data.columns:
        titanic_data.drop("zero", axis=1, inplace=True)

    print("Dataset after removing 'zero' column:")
    print(titanic_data)

    print("Values of 'Sex' column:")
    print(pd.get_dummies(titanic_data["Sex"]))

  
    sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)

    titanic_data = pd.concat([titanic_data, sex, pclass], axis=1)
    print("Dataset after concatenating new columns:")
    print(titanic_data)

    print("Dataset after removing irrelevant columns:")
    titanic_data.drop(["Sex", "SibSp", "Parch", "Embarked", "Pclass"], axis=1, inplace=True)
    print(titanic_data)

    X = titanic_data.drop("Survived", axis=1)
    Y = titanic_data["Survived"]

    # Step 4: Data Training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    logmodel = LogisticRegression(max_iter=200)
    logmodel.fit(X_train, Y_train)

    # Step 5: Data Testing
    predictions = logmodel.predict(X_test)

    # Step 6: Calculate Accuracy
    print("Classification Report:")
    print(classification_report(Y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))

    print("Accuracy Score:")
    print(accuracy_score(Y_test, predictions))

   # Optional Dataset file (command line var jo all dataset disat aahe toh ya file madhe save jhala)
    output_file = "AllEntriesOutput.csv"
    titanic_data.to_csv(output_file, index=False)
    print(f"All entries have been saved to '{output_file}'")

def main():
    print("_____ Marvellous Infosystems by Tanvi Ghodke _____")
    print("__ Supervised Machine Learning __")
    print("__ Logistic Regression on Titanic Dataset __")

    MarvellousTitanicLogistics()

if __name__ == "__main__":
    main()
