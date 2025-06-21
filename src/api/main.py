from fastapi import FastAPI

app = FastAPI(
    title="Anomaly Detector API"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Anomaly Detector API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}