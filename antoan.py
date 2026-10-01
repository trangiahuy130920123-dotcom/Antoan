import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import tempfile
import os

# =========================================================
# SafeStudent AI - Streamlit
# Không cần Google Colab. Model YOLO sẽ được tải tự động.
# =========================================================

st.set_page_config(
    page_title="SafeStudent AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- CSS --------------------
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container {max-width:1250px;padding-top:1rem;}
.hero{
    padding:24px;border-radius:20px;margin-bottom:18px;
    background:linear-gradient(135deg,#071521,#0a2638);
    border:1px solid #16455d;
}
.hero h1{margin:0;font-size:34px;}
.hero p{margin:7px 0 0;color:#9db6c2;}
.card{
    border:1px solid #20343f;border-radius:16px;padding:15px;
    background:#0b151c;text-align:center;min-height:105px;
}
.card h2{margin:0 0 5px;}
.safe{color:#22c55e;font-weight:700;}
.warn{color:#f59e0b;font-weight:700;}
.danger{color:#ef4444;font-weight:700;}
.info{color:#38bdf8;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# -------------------- Model --------------------
@st.cache_resource(show_spinner="Đang tải AI model lần đầu...")
def get_model():
    try:
        from ultralytics import YOLO
        # YOLO sẽ tự tải file nếu chưa có.
        return YOLO("yolo11n.pt"), None
    except Exception as e:
        return None, str(e)

# -------------------- Detection --------------------
def analyze(frame, model, conf, show_labels):
    result = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = result.plot(labels=show_labels, conf=show_labels)

    names = result.names
    labels = []

    if result.boxes is not None:
        for cls in result.boxes.cls.tolist():
            labels.append(str(names[int(cls)]))

    counts = Counter(labels)

    people = counts.get("person", 0)
    motorcycles = counts.get("motorcycle", 0)
    bicycles = counts.get("bicycle", 0)
    cars = counts.get("car", 0)
    phones = counts.get("cell phone", 0)
    traffic_lights = counts.get("traffic light", 0)

    warnings = []

    # Đây là cảnh báo hỗ trợ, không phải kết luận vi phạm.
    if phones:
        warnings.append(("danger", f"📱 Phát hiện {phones} điện thoại trong khung hình."))

    if people and motorcycles:
        warnings.append(("warn", "🛵 Có người và xe máy — cần kiểm tra hành vi giao thông."))

    if people and traffic_lights:
        warnings.append(("info", "🚦 Phát hiện người và đèn giao thông — có thể dùng để kiểm tra khu vực giao thông."))

    if not warnings:
        status = "🟢 CHƯA PHÁT HIỆN CẢNH BÁO"
        status_class = "safe"
    elif any(x[0] == "danger" for x in warnings):
        status = "🔴 CẢNH BÁO"
        status_class = "danger"
    else:
        status = "🟠 CẦN KIỂM TRA"
        status_class = "warn"

    return annotated, counts, status, status_class, warnings


# -------------------- Header --------------------
st.markdown("""
<div class="hero">
<h1>🚦 SafeStudent AI</h1>
<p>Hệ thống hỗ trợ nhận diện đối tượng giao thông an toàn cho học sinh</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Cài đặt AI")
    conf = st.slider("Độ tin cậy", 0.10, 0.90, 0.35, 0.05)
    show_labels = st.checkbox("Hiện nhãn trên ảnh/video", True)
    st.divider()
    st.caption("Model: YOLO11n")
    st.caption("Chạy trực tiếp bằng Streamlit")
    st.caption("Không cần Google Colab")

model, error = get_model()

if error:
    st.error("Không thể tải YOLO.")
    st.code("pip install ultralytics opencv-python pillow numpy")
    st.exception(Exception(error))
    st.stop()

tabs = st.tabs(["🖼️ Ảnh", "🎞️ Video", "📊 Hướng dẫn"])

# =========================================================
# IMAGE
# =========================================================
with tabs[0]:
    uploaded = st.file_uploader(
        "Chọn ảnh giao thông",
        type=["jpg", "jpeg", "png", "webp"],
        key="img"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)

        with st.spinner("🤖 AI đang phân tích..."):
            annotated, counts, status, status_class, warnings = analyze(
                frame, model, conf, show_labels
            )

        st.image(
            annotated,
            caption="Kết quả nhận diện",
            use_container_width=True
        )

        people = counts.get("person", 0)
        motorcycles = counts.get("motorcycle", 0)
        phones = counts.get("cell phone", 0)
        traffic_lights = counts.get("traffic light", 0)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(f'<div class="card"><h2>{people}</h2>👤 Người</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><h2>{motorcycles}</h2>🛵 Xe máy</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><h2>{phones}</h2>📱 Điện thoại</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="card"><h2>{traffic_lights}</h2>🚦 Đèn giao thông</div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="card"><div class="{status_class}">{status}</div></div>', unsafe_allow_html=True)

        st.subheader("📋 Phân tích")

        if warnings:
            for kind, message in warnings:
                if kind == "danger":
                    st.error(message)
                elif kind == "warn":
                    st.warning(message)
                else:
                    st.info(message)
        else:
            st.success("Chưa phát hiện đối tượng tạo cảnh báo theo model hiện tại.")

        st.info(
            "Bản YOLO mặc định không thể xác định chính xác đội/không đội mũ bảo hiểm. "
            "Muốn có chức năng đó cần model được huấn luyện riêng."
        )

# =========================================================
# VIDEO
# =========================================================
with tabs[1]:
    uploaded_video = st.file_uploader(
        "Chọn video giao thông",
        type=["mp4", "mov", "avi", "mkv"],
        key="video"
    )

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("▶️ Bắt đầu phân tích video", type="primary", use_container_width=True):
            suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_video.getbuffer())
                input_path = f.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Không thể mở video.")
                os.unlink(input_path)
                st.stop()

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Giới hạn xử lý khoảng 150 frame để demo trên Streamlit nhẹ hơn.
            max_frames = 150
            step = max(1, total // max_frames)

            out_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name

            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            progress = st.progress(0)
            preview = st.empty()
            all_counts = Counter()

            frame_index = 0
            processed = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    annotated, counts, _, _, _ = analyze(
                        frame, model, conf, show_labels
                    )
                    all_counts.update(counts)
                    writer.write(annotated)

                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Đang phân tích: {min(frame_index/total,1):.0%}",
                        use_container_width=True
                    )

                    processed += 1
                    progress.progress(min(processed / max_frames, 1.0))

                frame_index += 1

            cap.release()
            writer.release()

            progress.progress(1.0)
            preview.empty()

            st.success("✅ Phân tích video hoàn tất.")

            if os.path.exists(out_path):
                st.video(out_path)

                with open(out_path, "rb") as result_file:
                    st.download_button(
                        "⬇️ Tải video kết quả",
                        result_file,
                        file_name="safestudent_result.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("👤 Người", all_counts.get("person", 0))
            c2.metric("🛵 Xe máy", all_counts.get("motorcycle", 0))
            c3.metric("📱 Điện thoại", all_counts.get("cell phone", 0))
            c4.metric("🚦 Đèn giao thông", all_counts.get("traffic light", 0))

            os.unlink(input_path)

# =========================================================
# GUIDE
# =========================================================
with tabs[2]:
    st.subheader("🚀 Cách chạy")
    st.code("""pip install -r requirements.txt
streamlit run app.py""", language="bash")

    st.subheader("📱 Chạy trên Streamlit Cloud")
    st.markdown("""
1. Tạo repository GitHub.
2. Upload `app.py` và `requirements.txt`.
3. Deploy repository bằng Streamlit Community Cloud.
4. Mở đường link trên iPhone.
""")

    st.subheader("🤖 Model hiện tại")
    st.write(
        "YOLO11n được tải tự động. Bản này nhận diện các đối tượng trong dataset COCO "
        "như người, xe máy, ô tô, điện thoại và đèn giao thông."
    )

    st.subheader("🪖 Muốn nhận diện mũ bảo hiểm?")
    st.write(
        "Có thể thay `yolo11n.pt` bằng model riêng, ví dụ `best.pt`, sau khi có dataset "
        "helmet/no_helmet được gán nhãn. Không cần thay toàn bộ giao diện."
    )

st.divider()
st.caption(
    "SafeStudent AI • Prototype giáo dục • Kết quả AI chỉ mang tính hỗ trợ quan sát, "
    "không nên dùng để tự động kết luận hoặc xử phạt học sinh."
)
