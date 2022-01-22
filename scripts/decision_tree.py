import os
from datetime import datetime
from typing import Tuple, List

import click
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_dataframe(df: pd.DataFrame, features: List[str], label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean the dataframe and return a cleaned dataframe
    """
    # Drop the columns that are not needed
    columns = [*features, label] if label in df.columns else features
    df = df[columns]
    # Handle missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    # Drop the rows that have missing values
    df = df.dropna()

    # Convert the string values to numeric values
    if "Sex" in df.columns:
        df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    if "Embarked" in df.columns:
        df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
    return df


def train_decision_tree_classifier(x_train, y_train):
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf = clf.fit(x_train, y_train)
    return clf


@click.command()
@click.option("--train", default="./data/train.csv", help="Path to the train data")
@click.option("--test", default="./data/test.csv", help="Path to the test data")
@click.option("--predict", default="./predictions/", help="Path to the predictions directory")
def decision_tree(train, test, predict):
    features = ["Sex", "Age", "Pclass", "SibSp", "Parch", "Embarked"]
    label = "Survived"
    train_df = pd.read_csv(train)
    train_df = preprocess_dataframe(train_df, features, label)
    x, y = train_df[features].to_numpy(), train_df[label].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = train_decision_tree_classifier(x_train, y_train)
    print(clf.score(x_test, y_test))

    # make predictions on the kaggle test set
    test_df = pd.read_csv(test)
    test_df = preprocess_dataframe(test_df, ["PassengerId", *features], label)
    passenger_ids = test_df["PassengerId"].to_numpy()
    x_kaggle_test = test_df[features].to_numpy()
    y_pred = clf.predict(x_kaggle_test)
    print(y_pred[:10])

    # write the predictions to a file
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    csvfile = os.path.join(predict, f"decision_tree_{date_str}.csv")
    predictions_df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": y_pred})
    predictions_df.to_csv(csvfile, index=False, header=True)


if __name__ == "__main__":
    decision_tree()
