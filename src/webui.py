import gradio as gr
import requests
import time

API_URL = "http://app:8000"

# 학습 시작 함수
def start_training(n_estimators, max_depth, random_state):
    payload = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth != '' else None,
        "random_state": random_state
    }
    resp = requests.post(f"{API_URL}/train", json=payload)
    return resp.json().get("message", "요청 실패")

# 상태 조회 함수
def get_status():
    resp = requests.get(f"{API_URL}/status")
    data = resp.json()
    status = data.get("status", "unknown")
    detail = data.get("detail", None)
    if status == "done" and detail:
        # 주요 성능지표만 추출
        acc = detail.get("accuracy", None)
        f1 = detail.get("1", {}).get("f1-score", None)
        return f"상태: {status}\n정확도: {acc}\nF1-score(이상): {f1}\n전체 리포트: {detail}"
    elif status == "error":
        return f"상태: {status}\n에러: {detail}"
    else:
        return f"상태: {status}"

with gr.Blocks() as demo:
    gr.Markdown("""
    # NSL-KDD 이상탐지 학습 대시보드
    하이퍼파라미터를 입력하고 학습을 시작하세요.\n상태는 실시간으로 갱신됩니다.
    """)
    with gr.Row():
        n_estimators = gr.Number(label="n_estimators", value=100)
        max_depth = gr.Textbox(label="max_depth (빈칸=제한없음)", value="")
        random_state = gr.Number(label="random_state", value=42)
    start_btn = gr.Button("학습 시작")
    status_box = gr.Textbox(label="학습 상태", lines=8)

    def train_and_poll(n_estimators, max_depth, random_state):
        msg = start_training(n_estimators, max_depth, random_state)
        for _ in range(60):  # 최대 60초 대기
            status = get_status()
            if "done" in status or "error" in status:
                return status
            time.sleep(2)
        return get_status()

    start_btn.click(
        train_and_poll,
        inputs=[n_estimators, max_depth, random_state],
        outputs=status_box
    )
    gr.Markdown("---")
    gr.Markdown("상태만 새로고침하려면 아래 버튼을 누르세요.")
    refresh_btn = gr.Button("상태 새로고침")
    refresh_btn.click(get_status, inputs=[], outputs=status_box)

demo.launch()
