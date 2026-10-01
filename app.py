import streamlit as st

try:
    import cv2
except ModuleNotFoundError:
    st.error("Thiếu OpenCV. Hãy thêm `opencv-python-headless` vào requirements.txt rồi Reboot app.")
    st.stop()
import pandas as pd
import tempfile
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Traffic Safety",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.title {text-align:center; font-size:34px; font-weight:800;}
.box {padding:12px; border-radius:12px; border:1px solid #333; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚦 AI TRAFFIC SAFETY</div>', unsafe_allow_html=True)
st.caption("Nhận diện học sinh và phương tiện tham gia giao thông")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Không tải được YOLO. Hãy cài ultralytics và bảo đảm có Internet ở lần chạy đầu.")
    st.code("pip install streamlit ultralytics opencv-python pandas")
    st.stop()

with st.sidebar:
    st.header("⚙️ Cấu hình")
    confidence = st.slider("Độ tin cậy", 0.10, 0.95, 0.45, 0.05)
    process_every = st.slider("Xử lý mỗi N frame", 1, 5, 1)
    st.info("Model YOLO mặc định nhận diện người và phương tiện. Muốn nhận diện mũ bảo hiểm chính xác, thay bằng model YOLO đã huấn luyện helmet.")

uploaded = st.file_uploader(
    "📹 Chọn video giao thông",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded:
    data = uploaded.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(data)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    video_area = st.empty()
    status = st.empty()
    progress = st.progress(0)

    col1, col2, col3, col4 = st.columns(4)
    c_people = col1.empty()
    c_motor = col2.empty()
    c_bike = col3.empty()
    c_warning = col4.empty()

    logs = []
    frame_id = 0
    warning_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        if frame_id % process_every != 0:
            continue

        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            verbose=False
        )

        annotated = results[0].plot()

        people = 0
        motorcycles = 0
        bicycles = 0

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                name = model.names.get(cls, str(cls))

                if name == "person":
                    people += 1
                elif name == "motorcycle":
                    motorcycles += 1
                elif name == "bicycle":
                    bicycles += 1

        # Cảnh báo cơ bản: có người + phương tiện trong cùng khung hình.
        # Đây là cảnh báo hỗ trợ, không phải kết luận vi phạm.
        current_warning = people > 0 and (motorcycles > 0 or bicycles > 0)
        if current_warning:
            warning_count += 1
            logs.append({
                "Frame": frame_id,
                "Thời gian (s)": round(frame_id / fps, 2),
                "Cảnh báo": "Có người và phương tiện đang được theo dõi",
                "Mức": "CẦN CHÚ Ý"
            })

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_area.image(annotated_rgb, channels="RGB", use_container_width=True)

        c_people.metric("👤 Người", people)
        c_motor.metric("🏍️ Xe máy", motorcycles)
        c_bike.metric("🚲 Xe đạp", bicycles)
        c_warning.metric("⚠️ Cảnh báo", warning_count)

        status.write(
            f"Frame {frame_id}/{total_frames} • FPS video: {fps:.1f}"
        )
        progress.progress(min(frame_id / max(total_frames, 1), 1.0))

    cap.release()
    progress.empty()

    st.subheader("📋 Lịch sử cảnh báo")
    df = pd.DataFrame(logs)

    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Tải CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "traffic_safety_log.csv",
            "text/csv"
        )
    else:
        st.success("Không có cảnh báo nào trong video.")

else:
    st.info("Hãy tải một video giao thông lên để bắt đầu.")
    st.markdown("""
    ### Hệ thống hiện có
    - 👤 Nhận diện người
    - 🏍️ Nhận diện xe máy
    - 🚲 Nhận diện xe đạp
    - 🚗 Nhận diện ô tô
    - 🎯 Theo dõi đối tượng bằng YOLO Tracking
    - ⚠️ Cảnh báo hỗ trợ
    - 📊 Thống kê trực tiếp
    - 📥 Xuất lịch sử CSV

    **Lưu ý:** YOLO mặc định không thể xác định chính xác “học sinh” hay
    “không đội mũ bảo hiểm”. Để làm phần đó cần model được huấn luyện riêng.
    """)
