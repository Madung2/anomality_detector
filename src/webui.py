import gradio as gr
import requests
import time

API_URL = "http://api:8000"

# 학습 시작 함수
def start_training(n_estimators, max_depth, random_state):
    payload = {
        "n_estimators": n_estimators,
        "max_depth": None if max_depth == 0 else max_depth,
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
    gr.Markdown("""
    **n_estimators**: 사용할 결정트리(Decision Tree)의 개수입니다. 값이 클수록 예측이 안정적이지만 느려질 수 있습니다.
    
    **max_depth**: 각 트리의 최대 깊이입니다. 0이면 제한 없이 분기합니다. 값이 크면 과적합 위험이 있습니다.
    
    **random_state**: 랜덤성을 제어하는 시드 값입니다. 같은 값이면 결과가 항상 같습니다.
    """)
    with gr.Row():
        n_estimators = gr.Slider(label="n_estimators", minimum=10, maximum=500, value=100, step=10)
        max_depth = gr.Slider(label="max_depth (0=제한없음)", minimum=0, maximum=50, value=0, step=1)
        random_state = gr.Slider(label="random_state", minimum=0, maximum=10000, value=42, step=1)
    start_btn = gr.Button("학습 시작")
    status_box = gr.Textbox(label="학습 상태", lines=8, value=get_status, every=2)

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

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
