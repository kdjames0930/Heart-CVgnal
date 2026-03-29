"""
Heart CV-gnal — Streamlit Launcher
====================================
Flask + CV 백엔드를 백그라운드 스레드로 기동하고,
기존 index.html UI를 Streamlit 컴포넌트로 풀스크린 렌더링합니다.

CV 로직·영상 스트리밍·UI 모두 app.py / index.html 과 완전히 동일합니다.

Run:
    $env:PYTHONPATH="src"; streamlit run app_streamlit.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ── src/ 경로 추가 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# app.py 에서 Flask 앱·CV 루프·카메라 인덱스만 가져옴
# (import 시점에 서버가 시작되지 않음 — __main__ 블록이 실행되지 않으므로)
from app import app as _flask_app, _cv_loop, _CAMERA_INDEX  # noqa: E402

FLASK_PORT = 5001

# ── 백엔드는 프로세스당 1회만 시작 ─────────────────────────────────────────
_backend_lock    = threading.Lock()
_backend_started = False


def _run_flask() -> None:
    _flask_app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


def _ensure_backend() -> None:
    global _backend_started
    with _backend_lock:
        if not _backend_started:
            threading.Thread(target=_cv_loop,  daemon=True).start()
            threading.Thread(target=_run_flask, daemon=True).start()
            _backend_started = True
            time.sleep(1.2)   # Flask 바인딩 대기


_ensure_backend()

# ── index.html 읽어서 URL만 절대 경로로 패치 ───────────────────────────────
# iframe 안에서 상대 경로 fetch·img src 가 작동하려면 Flask 주소가 필요
_BASE = f"http://localhost:{FLASK_PORT}"
_html = (
    (Path(__file__).parent / "templates" / "index.html")
    .read_text(encoding="utf-8")
    .replace('src="/video_feed"',  f'src="{_BASE}/video_feed"')
    .replace("fetch('/status')",   f"fetch('{_BASE}/status')")
)

# ── Streamlit 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart CV-gnal ♥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Streamlit 고유 UI(헤더·푸터·여백) 숨기고 iframe 풀스크린으로
st.markdown(
    """
    <style>
      #MainMenu,
      header[data-testid="stHeader"],
      footer,
      [data-testid="stToolbar"]       { display: none !important; }

      .main .block-container {
        padding:   0 !important;
        max-width: 100% !important;
      }
      section[data-testid="stMain"] > div:first-child {
        padding: 0 !important;
      }

      /* iframe 풀스크린 */
      iframe {
        width:   100vw   !important;
        height:  100vh   !important;
        border:  none    !important;
        display: block   !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 기존 UI 렌더링 ────────────────────────────────────────────────────────
components.html(_html, height=950, scrolling=False)
