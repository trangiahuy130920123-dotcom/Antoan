import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import tempfile
import os

st.set_page_config(
    page_title="SafeStudent AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.stApp {background:#f7fbff;}
.block-container {max-width:820px;padding:18px 14px 100px;}

.hero{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:58px;height:58px;border-radius:17px;background:linear-gradient(145deg,#1677ed,#65b8ff);
display:flex;align-items:center;justify-content:center;font-size:30px;box-shadow:0 7px 18px #0073ff22}
.brand h1{margin:0;color:#102a4c;font-size:28px;font-weight:800;line-height:1.05}
.brand h1 span{color:#1677ed}
.brand p{margin:6px 0 0;color:#60758c;font-size:13px}
.live{white-space:nowrap;padding:9px 12px;border-radius:20px;background:#e3faef;color:#16a765;font-weight:700;font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#13b86b;margin-right:6px}

.banner{padding:17px 18px;border:1px solid #b9dbff;border-radius:17px;background:#edf7ff;margin-bottom:16px}
.banner-title{font-size:18px;color:#112b4d;font-weight:800;margin-bottom:5px}
.banner-text{color:#5c7187;font-size:14px}

.card{border-radius:17px;padding:16px;min-height:140px;border:1px solid #d9e7f3;margin-bottom:14px;
box-shadow:0 3px 12px #19466e0b}
.blue{background:#f2f8ff;border-color:#cfe5ff}
.red{background:#fff5f5;border-color:#ffd3d3}
.purple{background:#f8f5ff;border-color:#ddd5ff}
.yellow{background:#fffaf0;border-color:#ffe3a6}
.green{background:#f1fff8;border-color:#c5f0da}
.icon{font-size:30px;margin-bottom:7px}
.title{color:#173250;font-size:15px;font-weight:700}
.value{color:#102a4c;font-size:39px;line-height:1;font-weight:800;margin:7px 0 9px}
.ok{color:#12a967;font-size:13px;font-weight:700}
.bad{color:#ed3e3e;font-size:13px;font-weight:700}
.muted{color:#718399;font-size:13px}

.section{background:#fff;border:1px solid #e0ebf4;border-radius:18px;padding:18px;margin:4px 0 15px;
box-shadow:0 3px 12px #19466e0a}
.section-title{color:#142f51;font-size:19px;font-weight:800;margin-bottom:14px}
.donutrow{display:flex;align-items:center;gap:22px}
.donut{width:145px;height:145px;border-radius:50%;position:relative;flex:none}
.donut:after{content:"";position:absolute;inset:26px;background:#fff;border-radius:50%}
.dc{position:absolute;z-index:2;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
color:#173250;font-weight:800;font-size:26px}
.dc small{font-size:13px;font-weight:500;color:#6d8095}
.legend{color:#435a72;font-size:14px;line-height:2}
.ld{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px}
.warning{padding:17px;border:1px solid #ffcaca;border-radius:17px;background:#fff5f5;margin-bottom:16px}
.wt{color:#df2626;font-weight:800;font-size:17px}
.wx{color:#52677e;font-size:14px;margin-top:5px}
.foot{padding:12px 14px;border-radius:13px;background:#edf5fb;color:#6c7f92;font-size:12px;text-align:center}
.nav{position:fixed;left:0;right:0;bottom:0;z-index:99;background:#fffffff7;border-top:1px solid #dfe9f2;
padding:8px 2px;display:flex;justify-content:center;backdrop-filter:blur(10px)}
.ni{width:20%;text-align:center;color:#6c7d8f;font-size:11px}
.na{color:#1477ed;font-weight:800}
.ni b{display:block;font-size:20px;margin-bottom:2px}

@media(max-width:600px){
.brand h1{font-size:25px}
.brand p{font-size:11px}
.live{font-size:10px;padding:7px 9px}
.logo{width:52px;height:52px}
.donut{width:125px;height:125px}
.donutrow{gap:14px}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang tải AI model...")
def load_model():
    try:
        from ultralytics import YOLO
        custom = Path("best.pt")
        if custom.exists():
            return YOLO(str(custom)), True, None
        return YOLO("yolo11n.pt"), False, None
    except Exception as e:
        return None, False, str(e)


model, is_custom, model_error = load_model()

if model_error:
    st.error("Không tải được AI model.")
    st.code("pip install streamlit ultralytics opencv-python-headless pillow numpy")
    st.error(str(model_error))
    st.stop()


def normalize_name(name):
    return str(name).strip().lower().replace("-", "_")


def count_classes(result):
    counts = Counter()
    if result.boxes is not None:
        for cls in result.boxes.cls.tolist():
            if isinstance(result.names, dict):
                name = result.names[int(cls)]
            else:
                name = result.names[int(cls)]
            counts[normalize_name(name)] += 1
    return counts


def count_matching(counts, keywords):
    total = 0
    for name, amount in counts.items():
        if any(word in name for word in keywords):
            total += amount
    return total


def analyze(frame):
    result = model.predict(frame, conf=conf, verbose=False)[0]
    annotated = result.plot(labels=True, conf=True)
    counts = count_classes(result)

    # Chỉ tính "học sinh" nếu model có class student/hoc_sinh.
    student = count_matching(
        counts,
        ["student", "hoc_sinh", "hoc sinh", "school_student"]
    )

    # Không dùng "person" làm học sinh.
    person = count_matching(counts, ["person"])

    stats = {
        "helmet": count_matching(
            counts, ["helmet", "with_helmet", "doi_mu"]
        ),
        "no_helmet": count_matching(
            counts,
            ["no_helmet", "no helmet", "without_helmet",
             "without helmet", "khong_mu", "khong mu", "khong_doi_mu"]
        ),
        "phone": count_matching(
            counts, ["cell_phone", "cell phone", "phone", "dien_thoai"]
        ),
        "motorcycle": count_matching(
            counts, ["motorcycle", "motorbike", "xe_may"]
        ),
        "student": student,
        "person": person,
        "traffic": count_matching(
            counts, ["traffic_light", "traffic light", "den_giao_thong"]
        ),
    }

    return annotated, stats


def dashboard(s):
    helmet = s["helmet"]
    no_helmet = s["no_helmet"]
    total = helmet + no_helmet
    pct = round(helmet / total * 100) if total else 0

    def card(cls, icon, title, value, message, message_class):
        st.markdown(
            f"""
            <div class="card {cls}">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="value">{value}</div>
                <div class="{message_class}">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)

    with left:
        card(
            "blue", "🪖", "Đội mũ bảo hiểm", helmet,
            "✓ Đã nhận diện" if helmet else "Chưa có dữ liệu",
            "ok" if helmet else "muted"
        )
    with right:
        card(
            "red", "🚫", "Không đội mũ bảo hiểm", no_helmet,
            "⚠ Cần kiểm tra" if no_helmet else "✓ Chưa phát hiện",
            "bad" if no_helmet else "muted"
        )

    with left:
        card(
            "purple", "📱", "Sử dụng điện thoại", s["phone"],
            "⚠ Cảnh báo" if s["phone"] else "✓ Không phát hiện",
            "bad" if s["phone"] else "ok"
        )
    with right:
        card(
            "yellow", "🛵", "Xe máy", s["motorcycle"],
            "Đối tượng nhận diện", "muted"
        )

    with left:
        card(
            "green", "👨‍🎓", "Học sinh", s["student"],
            "Chỉ tính class student", "muted"
        )
    with right:
        card(
            "blue", "🚦", "Đèn giao thông", s["traffic"],
            "Đối tượng nhận diện", "muted"
        )

    st.markdown(
        '<div class="section"><div class="section-title">📊 Tỷ lệ an toàn giao thông</div>',
        unsafe_allow_html=True,
    )

    if total:
        deg = pct * 3.6
        st.markdown(
            f"""
            <div class="donutrow">
                <div class="donut"
                     style="background:conic-gradient(#10b968 0deg {deg}deg,#ff5a61 {deg}deg 360deg)">
                    <div class="dc">{pct}%<small>An toàn</small></div>
                </div>
                <div class="legend">
                    <div>
                        <span class="ld" style="background:#10b968"></span>
                        Đội mũ bảo hiểm <b>{helmet}</b> ({pct}%)
                    </div>
                    <div>
                        <span class="ld" style="background:#ff5a61"></span>
                        Không đội mũ <b>{no_helmet}</b> ({100-pct}%)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Chưa có dữ liệu mũ bảo hiểm. Model cần class "
            "helmet/no_helmet để nhận diện thật."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if no_helmet:
        st.markdown(
            f"""
            <div class="warning">
                <div class="wt">⚠️ Cảnh báo</div>
                <div class="wx">
                    Phát hiện <b>{no_helmet}</b> trường hợp không đội mũ bảo hiểm.
                    Vui lòng nhắc nhở và tuân thủ quy định an toàn giao thông!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif s["phone"]:
        st.markdown(
            f"""
            <div class="warning">
                <div class="wt">⚠️ Cảnh báo</div>
                <div class="wx">
                    Phát hiện <b>{s["phone"]}</b> trường hợp có điện thoại.
                    Cần kiểm tra hành vi sử dụng điện thoại khi tham gia giao thông.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not is_custom:
        st.info(
            "💡 Model YOLO mặc định không phân biệt học sinh với người ngoài "
            "và không có class mũ bảo hiểm. Để nhận diện đúng học sinh + "
            "helmet/no_helmet, đặt model best.pt đã train riêng cạnh app.py."
        )


st.markdown(
    """
    <div class="hero">
        <div class="brand">
            <div class="logo">🛡️</div>
            <div>
                <h1>SafeStudent <span>AI</span></h1>
                <p>Nhận diện hành vi học sinh tham gia giao thông an toàn</p>
            </div>
        </div>
        <div class="live"><span class="dot"></span>Đang hoạt động</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
        <div class="banner-title">📹 Phân tích video hoặc ảnh!</div>
        <div class="banner-text">
            AI sẽ nhận diện và thống kê các đối tượng trong hình ảnh/video.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("⚙️ Cài đặt AI"):
    conf = st.slider("Độ tin cậy", 0.10, 0.90, 0.35, 0.05)
    st.caption(
        "Model: " + ("best.pt — model riêng" if is_custom else "YOLO11n — mặc định")
    )

if "stats" not in st.session_state:
    st.session_state.stats = {
        "helmet": 0,
        "no_helmet": 0,
        "phone": 0,
        "motorcycle": 0,
        "student": 0,
        "person": 0,
        "traffic": 0,
    }

home_tab, image_tab, video_tab = st.tabs(
    ["🏠 Tổng quan", "🖼️ Ảnh", "🎥 Video"]
)

with home_tab:
    dashboard(st.session_state.stats)

with image_tab:
    uploaded = st.file_uploader(
        "🖼️ Chọn ảnh giao thông",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded:
        frame = np.array(Image.open(uploaded).convert("RGB"))

        with st.spinner("🤖 Đang phân tích..."):
            annotated, stats = analyze(frame)

        st.image(annotated, use_container_width=True)
        st.session_state.stats = stats
        st.success("✅ Phân tích ảnh hoàn tất.")
        dashboard(stats)

with video_tab:
    uploaded_video = st.file_uploader(
        "🎥 Chọn video giao thông",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if uploaded_video:
        st.video(uploaded_video)

        if st.button(
            "▶️ Phân tích video",
            type="primary",
            use_container_width=True,
        ):
            suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                input_path = temp_file.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Không thể mở video.")
                os.unlink(input_path)
                st.stop()

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            progress = st.progress(0.0)
            preview = st.empty()
            combined = Counter()

            max_frames = 150
            step = max(1, total // max_frames)
            frame_index = 0
            processed = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    annotated, stats = analyze(frame)
                    combined.update(stats)
                    writer.write(annotated)

                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                    processed += 1
                    progress.progress(
                        min(processed / max_frames, 1.0)
                    )

                frame_index += 1

            cap.release()
            writer.release()
            preview.empty()
            progress.progress(1.0)

            final_stats = {
                "helmet": combined["helmet"],
                "no_helmet": combined["no_helmet"],
                "phone": combined["phone"],
                "motorcycle": combined["motorcycle"],
                "student": combined["student"],
                "person": combined["person"],
                "traffic": combined["traffic"],
            }

            st.session_state.stats = final_stats

            st.success("✅ Phân tích video hoàn tất.")
            st.video(output_path)

            with open(output_path, "rb") as result_file:
                st.download_button(
                    "⬇️ Tải video kết quả",
                    result_file,
                    file_name="safestudent_result.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            dashboard(final_stats)
            os.unlink(input_path)

st.markdown(
    """
    <div class="foot">
        🛡️ SafeStudent AI • Vì một môi trường học đường an toàn hơn
    </div>
    <div class="nav">
        <div class="ni na"><b>⌂</b>Trang chủ</div>
        <div class="ni"><b>▧</b>Ảnh</div>
        <div class="ni"><b>▣</b>Video</div>
        <div class="ni"><b>▥</b>Thống kê</div>
        <div class="ni"><b>⚙</b>Cài đặt</div>
    </div>
    """,
    unsafe_allow_html=True,
)
