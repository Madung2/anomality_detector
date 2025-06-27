import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data.KDDTrain_columns import KDDTRAIN_PLUS_COLUMNS

class AnomalyDetector:
    """네트워크 트래픽의 이상을 탐지하는 클래스"""
    def __init__(self, columns: list = KDDTRAIN_PLUS_COLUMNS, model_path: str = None,
                 n_estimators: int = 100, max_depth: int = None, random_state: int = 42) -> None:
        """
        Args:
            columns (list): 데이터 컬럼명 리스트
            model_path (str, optional): 저장된 모델 경로 (선택사항)
            n_estimators (int): 랜덤포레스트 트리 개수
            max_depth (int): 트리 최대 깊이
            random_state (int): 랜덤 시드
        """
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.columns = columns
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        if model_path:
            self.load(model_path)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """데이터를 데이터 프레임으로 로드하는 함수
        
        Args:
            file_path (str): 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        df = pd.read_csv(file_path, names=self.columns)
        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """데이터 전처리를 수행하는 함수
        
        Args:
            df (pd.DataFrame): 전처리할 DataFrame
            fit (bool): 새로운 데이터에 맞춰 인코더와 스케일러를 학습할지 여부
            
        Returns:
            tuple[np.ndarray, np.ndarray]: (전처리된 X 데이터, y 라벨)
        """
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

    def train(self, X: np.ndarray, y: np.ndarray, n_estimators: int = None, max_depth: int = None, random_state: int = None) -> None:
        """모델을 학습시키는 함수 (랜덤 포레스트 모델)
        Args:
            X (np.ndarray): 학습 데이터
            y (np.ndarray): 라벨 데이터
            n_estimators (int, optional): 트리 개수 (없으면 생성자 값 사용)
            max_depth (int, optional): 트리 최대 깊이 (없으면 생성자 값 사용)
            random_state (int, optional): 랜덤 시드 (없으면 생성자 값 사용)
        """
        n_estimators = n_estimators if n_estimators is not None else self.n_estimators
        max_depth = max_depth if max_depth is not None else self.max_depth
        random_state = random_state if random_state is not None else self.random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측을 수행하는 함수
        
        Args:
            X (np.ndarray): 예측할 데이터
            
        Returns:
            np.ndarray: 예측 결과 (0: 정상, 1: 이상)
            
        Raises:
            ValueError: 모델이 학습되지 않은 경우
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        return self.model.predict(X)

    def save(self, model_dir: str) -> None:
        """모델을 저장하는 함수
        
        Args:
            model_dir (str): 모델을 저장할 디렉토리 경로
        """
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'rf_model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))

    def load(self, model_dir):
        """저장된 모델을 불러오는 함수
        
        Args:
            model_dir: 모델이 저장된 디렉토리 경로
        """
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