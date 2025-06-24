import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data.KDDTrain_columns import KDDTRAIN_PLUS_COLUMNS

class AnomalyDetector:
    def __init__(self, columns=KDDTRAIN_PLUS_COLUMNS, model_path=None):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.columns = columns
        if model_path:
            self.load(model_path)

    def load_data(self, file_path):
        df = pd.read_csv(file_path, names=self.columns)
        return df

    def preprocess(self, df, fit=True):
        df = df.copy()
        # 이진 라벨링
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        # 범주형 인코딩
        for col in ['protocol_type', 'service', 'flag']:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col])
        X = df.drop('label', axis=1)
        y = df['label']
        # 스케일링
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        return self.model.predict(X)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'rf_model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))

    def load(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.joblib'))

if __name__ == "__main__":
    file_path = './data/KDDTrain+.txt'
    model_dir = './saved_models'
    detector = AnomalyDetector()
    df = detector.load_data(file_path)
    X, y = detector.preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    detector.train(X_train, y_train)
    detector.save(model_dir)
    y_pred = detector.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred)) 