from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, n_components = 5):
        dataset = fetch_covtype()

        self.X = dataset.data
        self.y = dataset.target
        self.feature_names = dataset.feature_names

        self.n_components = n_components
        self.mean = None

        self.components = None
        self.scaler = StandardScaler()

    
    def split(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            random_state=42,
            stratify=self.y
        )

        return X_train, X_test, y_train, y_test

    def fit_dim_reduction(self, X_train):
        self.mean = np.mean(
            X_train,
            axis=0
        )
        X_centered = X_train - self.mean

        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = vt[
            :self.n_components
        ].T

    def transform_dim_reduced(self, X):
        X_centered = X - self.mean
        X_reduced = (
            X_centered
            @ self.components
        )
        return X_reduced

    def fit_transform_dim_reduction(self, X_train):
        self.fit_dim_reduction(X_train)

        return self.transform_dim_reduced(
            X_train
        )  


    def prepare_data(self):
        X_train, X_test, y_train, y_test = self.split()

        X_train = self.fit_transform_dim_reduction(X_train)

        X_test = self.transform_dim_reduced(
            X_test
        )

        X_train = self.scaler.fit_transform(
            X_train
        )

        X_test = self.scaler.transform(
            X_test
        )
        y_train = np.eye(7)[y_train.astype(int) - 1].T
        y_test = np.eye(7)[y_test.astype(int) - 1].T

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


    def getdataframe(self):
        df = pd.DataFrame(
            self.X,
            columns=self.feature_names
        )
        df['target'] = self.y

        return df



