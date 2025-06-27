from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import threading
import time
import sys
sys.path.append("../../src/train")
from src.train.model import AnomalyDetector
import redis
import json

app = FastAPI(
    title="Anomaly Detector API"
)

# Redis 클라이언트 연결 (기본: localhost, 6379)
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
REDIS_KEY = 'anomaly_train_status'

# 상태 저장/조회 함수
def set_train_status(status_dict):
    redis_client.set(REDIS_KEY, json.dumps(status_dict))

def get_train_status():
    val = redis_client.get(REDIS_KEY)
    if val:
        return json.loads(val)
    return {"status": "idle", "detail": None}

class TrainParams(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = 42

# 비동기 학습 함수
def train_model(params: TrainParams):
    try:
        set_train_status({"status": "training", "detail": None})
        file_path = './data/KDDTrain+.txt'
        model_dir = './saved_models'
        detector = AnomalyDetector(n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=params.random_state)
        df = detector.load_data(file_path)
        X, y = detector.preprocess(df, fit=True)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params.random_state)
        detector.train(X_train, y_train)
        detector.save(model_dir)
        y_pred = detector.predict(X_test)
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        set_train_status({"status": "done", "detail": report})
    except Exception as e:
        set_train_status({"status": "error", "detail": str(e)})

@app.get("/")
def read_root():
    return {"message": "Welcome to the Anomaly Detector API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/train")
def start_training(params: TrainParams):
    status = get_train_status()
    if status["status"] == "training":
        return {"message": "이미 학습이 진행 중입니다."}
    thread = threading.Thread(target=train_model, args=(params,))
    thread.start()
    return {"message": "학습을 시작합니다.", "params": params.dict()}

@app.get("/status")
def get_status():
    return get_train_status()