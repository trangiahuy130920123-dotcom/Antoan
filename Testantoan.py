import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import tempfile, os

st.set_page_config(page_title='SafeStudent AI', page_icon='🛡️', layout='centered', initial_sidebar_state='collapsed')

st.markdown(r'''<style>
#MainMenu,footer,header{visibility:hidden}.stApp{background:#f7fbff}.block-container{max-width:820px;padding:18px 14px 100px}
.hero{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:18px}.brand{display:flex;align-items:center;gap:12px}.logo{width:58px;height:58px;border-radius:17px;background:linear-gradient(145deg,#1677ed,#65b8ff);display:flex;align-items:center;justify-content:center;font-size:30px;box-shadow:0 7px 18px #0073ff22}.brand h1{margin:0;color:#102a4c;font-size:28px;font-weight:800;line-height:1.05}.brand h1 span{color:#1677ed}.brand p{margin:6px 0 0;color:#60758c;font-size:13px}.live{white-space:nowrap;padding:9px 12px;border-radius:20px;background:#e3faef;color:#16a765;font-weight:700;font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#13b86b;margin-right:6px}
.banner{padding:17px 18px;border:1px solid #b9dbff;border-radius:17px;background:#edf7ff;margin-bottom:16px}.banner-title{font-size:18px;color:#112b4d;font-weight:800;margin-bottom:5px}.banner-text{color:#5c7187;font-size:14px}
.card{border-radius:17px;padding:16px;min-height:140px;border:1px solid #d9e7f3;margin-bottom:14px;box-shadow:0 3px 12px #19466e0b}.blue{background:#f2f8ff;border-color:#cfe5ff}.red{background:#fff5f5;border-color:#ffd3d3}.purple{background:#f8f5ff;border-color:#ddd5ff}.yellow{background:#fffaf0;border-color:#ffe3a6}.green{background:#f1fff8;border-color:#c5f0da}.icon{font-size:30px;margin-bottom:7px}.title{color:#173250;font-size:15px;font-weight:700}.value{color:#102a4c;font-size:39px;line-height:1;font-weight:800;margin:7px 0 9px}.ok{color:#12a967;font-size:13px;font-weight:700}.bad{color:#ed3e3e;font-size:13px;font-weight:700}.muted{color:#718399;font-size:13px}
.section{background:#fff;border:1px solid #e0ebf4;border-radius:18px;padding:18px;margin:4px 0 15px;box-shadow:0 3px 12px #19466e0a}.section-title{color:#142f51;font-size:19px;font-weight:800;margin-bottom:14px}.donutrow{display:flex;align-items:center;gap:22px}.donut{width:145px;height:145px;border-radius:50%;position:relative;flex:none}.donut:after{content:'';position:absolute;inset:26px;background:#fff;border-radius:50%}.dc{position:absolute;z-index:2;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#173250;font-weight:800;font-size:26px}.dc small{font-size:13px;font-weight:500;color:#6d8095}.legend{color:#435a72;font-size:14px;line-height:2}.ld{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px}.warning{padding:17px;border:1px solid #ffcaca;border-radius:17px;background:#fff5f5;margin-bottom:16px}.wt{color:#df2626;font-weight:800;font-size:17px}.wx{color:#52677e;font-size:14px;margin-top:5px}.foot{padding:12px 14px;border-radius:13px;background:#edf5fb;color:#6c7f92;font-size:12px;text-align:center}.nav{position:fixed;left:0;right:0;bottom:0;z-index:99;background:#fffffff7;border-top:1px solid #dfe9f2;padding:8px 2px;display:flex;justify-content:center;backdrop-filter:blur(10px)}.ni{width:20%;text-align:center;color:#6c7d8f;font-size:11px}.na{color:#1477ed;font-weight:800}.ni b{display:block;font-size:20px;margin-bottom:2px}
@media(max-width:600px){.brand h1{font-size:25px}.brand p{font-size:11px}.live{font-size:10px;padding:7px 9px}.logo{width:52px;height:52px}.donut{width:125px;height:125px}.donutrow{gap:14px}}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Đang tải AI model...')
def load_model():
    try:
        from ultralytics import YOLO
        custom=Path('best.pt')
        return YOLO(str(custom)) if custom.exists() else YOLO('yolo11n.pt'), custom.exists(), None
    except Exception as e:return None,False,str(e)

model,is_custom,err=load_model()
if err:
    st.error('Không tải được AI model.');st.code('pip install streamlit ultralytics opencv-python-headless pillow numpy');st.stop()

with st.expander('⚙️ Cài đặt AI'):
    conf=st.slider('Độ tin cậy',.1,.9,.35,.05)
    st.caption('Model: '+('best.pt — model riêng' if is_custom else 'YOLO11n — mặc định'))
    if is_custom: st.caption('🎓 Chế độ: chỉ thống kê class student, không gom người ngoài vào học sinh.')

st.markdown('''<div class="hero"><div class="brand"><div class="logo">🛡️</div><div><h1>SafeStudent <span>AI</span></h1><p>Nhận diện hành vi học sinh tham gia giao thông an toàn</p></div></div><div class="live"><span class="dot"></span>Đang hoạt động</div></div>''',unsafe_allow_html=True)
st.markdown('''<div class="banner"><div class="banner-title">📹 Phân tích video hoặc ảnh!</div><div class="banner-text">AI sẽ nhận diện và thống kê các đối tượng trong hình ảnh/video.</div></div>''',unsafe_allow_html=True)

if 'stats' not in st.session_state: st.session_state.stats={'helmet':0,'no_helmet':0,'phone':0,'motorcycle':0,'student':0,'person':0,'traffic':0}

def analyze(frame):
    r=model.predict(frame,conf=conf,verbose=False)[0]; ann=r.plot(labels=True,conf=True); labels=[]
    if r.boxes is not None:
        for c in r.boxes.cls.tolist(): labels.append(str(r.names[int(c)]).lower())
    c=Counter(labels)
    def n(words): return sum(v for k,v in c.items() if any(w in k for w in words))

    # Chỉ tính học sinh khi model có class student/hoc sinh.
    # Không tự coi mọi người đi đường là học sinh.
    student_count = n(['student','hoc sinh','student_uniform','school student'])
    person_count = n(['person'])
    if is_custom:
        person_for_dashboard = student_count
    else:
        person_for_dashboard = 0

    return ann,{'helmet':n(['helmet','with helmet','doi mu']),'no_helmet':n(['no helmet','without helmet','no_helmet','khong doi mu','khong mu']),'phone':n(['cell phone','phone','dien thoai']),'motorcycle':n(['motorcycle','motorbike','xe may']),'student':student_count,'person':person_for_dashboard,'traffic':n(['traffic light','den giao thong'])}

def dashboard(s):
    h,nh=s['helmet'],s['no_helmet']; total=h+nh; pct=round(h/total*100) if total else 0
    def card(cls,ico,title,val,msg,mc):
        st.markdown(f'<div class="card {cls}"><div class="icon">{ico}</div><div class="title">{title}</div><div class="value">{val}</div><div class="{mc}">{msg}</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a: card('blue','🪖','Đội mũ bảo hiểm',h,'✓ Đã nhận diện' if h else 'Chưa có dữ liệu','ok' if h else 'muted')
    with b: card('red','🚫','Không đội mũ bảo hiểm',nh,'⚠ Cần kiểm tra' if nh else '✓ Chưa phát hiện','bad' if nh else 'muted')
    with a: card('purple','📱','Sử dụng điện thoại',s['phone'],'⚠ Cảnh báo' if s['phone'] else '✓ Không phát hiện','bad' if s['phone'] else 'ok')
    with b: card('yellow','🛵','Xe máy',s['motorcycle'],'Đối tượng nhận diện','muted')
    with a: card('green','👤','Học sinh',s['student'],'Học sinh được nhận diện','muted')
    with b: card('blue','🚦','Đèn giao thông',s['traffic'],'Đối tượng nhận diện','muted')
    st.markdown('<div class="section"><div class="section-title">📊 Tỷ lệ an toàn giao thông</div>',unsafe_allow_html=True)
    if total:
        deg=pct*3.6
        st.markdown(f'''<div class="donutrow"><div class="donut" style="background:conic-gradient(#10b968 0deg {deg}deg,#ff5a61 {deg}deg 360deg)"><div class="dc">{pct}%<small>An toàn</small></div></div><div class="legend"><div><span class="ld" style="background:#10b968"></span>Đội mũ bảo hiểm <b>{h}</b> ({pct}%)</div><div><span class="ld" style="background:#ff5a61"></span>Không đội mũ <b>{nh}</b> ({100-pct}%)</div></div></div>''',unsafe_allow_html=True)
    else: st.info('Chưa có dữ liệu mũ bảo hiểm. Model cần class helmet/no_helmet để nhận diện thật.')
    st.markdown('</div>',unsafe_allow_html=True)
    if nh: st.markdown(f'<div class="warning"><div class="wt">⚠️ Cảnh báo</div><div class="wx">Phát hiện <b>{nh}</b> trường hợp không đội mũ bảo hiểm. Vui lòng nhắc nhở và tuân thủ quy định an toàn giao thông!</div></div>',unsafe_allow_html=True)
    elif s['phone']: st.markdown(f'<div class="warning"><div class="wt">⚠️ Cảnh báo</div><div class="wx">Phát hiện <b>{s["phone"]}</b> trường hợp có điện thoại. Cần kiểm tra hành vi sử dụng điện thoại khi tham gia giao thông.</div></div>',unsafe_allow_html=True)
    if not is_custom:
        st.warning('🎓 Bản demo mặc định chưa phân biệt học sinh với người ngoài. Để chỉ nhận diện học sinh, đặt model best.pt có class `student` (và helmet/no_helmet) cạnh app.py.')

home,imgtab,vtab=st.tabs(['🏠 Tổng quan','🖼️ Ảnh','🎥 Video'])
with home: dashboard(st.session_state.stats)
with imgtab:
    up=st.file_uploader('🖼️ Chọn ảnh giao thông',type=['jpg','jpeg','png','webp'])
    if up:
        frame=np.array(Image.open(up).convert('RGB'))
        with st.spinner('🤖 Đang phân tích...'): ann,s=analyze(frame)
        st.image(ann,use_container_width=True);st.session_state.stats=s;st.success('✅ Phân tích ảnh hoàn tất.');dashboard(s)
with vtab:
    up=st.file_uploader('🎥 Chọn video giao thông',type=['mp4','mov','avi','mkv'])
    if up:
        st.video(up)
        if st.button('▶️ Phân tích video',type='primary',use_container_width=True):
            suf=Path(up.name).suffix or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False,suffix=suf) as f:f.write(up.getbuffer());inp=f.name
            cap=cv2.VideoCapture(inp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1;fps=cap.get(cv2.CAP_PROP_FPS) or 25;w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));hh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));out=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name
            wr=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,hh));pr=st.progress(0);prev=st.empty();comb=Counter();step=max(1,total//150);i=0;done=0
            while True:
                ok,fr=cap.read()
                if not ok:break
                if i%step==0:
                    ann,s=analyze(fr);comb.update(s);wr.write(ann);prev.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True);done+=1;pr.progress(min(done/max(1,min(150,total)),1.0))
                i+=1
            cap.release();wr.release();prev.empty();pr.progress(1.0);s=dict(comb);st.session_state.stats=s;st.success('✅ Phân tích video hoàn tất.');st.video(out)
            with open(out,'rb') as f: st.download_button('⬇️ Tải video kết quả',f,'safestudent_result.mp4','video/mp4',use_container_width=True)
            dashboard(s);os.unlink(inp)

st.markdown('''<div class="foot">🛡️ SafeStudent AI • Vì một môi trường học đường an toàn hơn</div><div class="nav"><div class="ni na"><b>⌂</b>Trang chủ</div><div class="ni"><b>▧</b>Ảnh</div><div class="ni"><b>▣</b>Video</div><div class="ni"><b>▥</b>Thống kê</div><div class="ni"><b>⚙</b>Cài đặt</div></div>''',unsafe_allow_html=True)
'''
print('written',p)
