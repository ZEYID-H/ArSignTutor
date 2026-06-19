from pathlib import Path
import sys, json, random, time
from collections import deque, Counter

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np, cv2, av
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path: sys.path.insert(0, str(APP_DIR))
import config

MODEL_PATH = next((p for p in [
    APP_DIR / "arsigntutor_final_results/best_rgb_domain_adapted_mobilenetv2.pth",
    APP_DIR / "arsigntutor_final_results/final_project_package/results_clean_training/rgb_mobilenetv2_training/best_rgb_domain_adapted_mobilenetv2.pth",
    APP_DIR / "best_rgb_domain_adapted_mobilenetv2.pth",
] if p.exists()), None)
CLASS_NAMES_JSON = next((p for p in [
    APP_DIR / "arsigntutor_final_results/class_names.json",
    APP_DIR / "class_names.json",
] if p.exists()), None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTHING=25; MIN_VOTES=15; CONF_OK=70.0; CONF_FRAME=50.0
PTS=10; STREAK_B=5; CD_OK=2.0; CD_WRONG=1.5; INFER_N=3; BLUR_K=21
RTC_CONFIG = {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]},{"urls":["stun:stun1.l.google.com:19302"]}]}

ARABIC = {"ain":"ع","al":"ال","aleff":"ا","bb":"ب","dal":"د","dha":"ظ","dhad":"ض","fa":"ف",
          "gaaf":"ق","ghain":"غ","ha":"هـ","haa":"ح","jeem":"ج","kaaf":"ك","khaa":"خ","la":"لا",
          "laam":"ل","meem":"م","nun":"ن","ra":"ر","saad":"ص","seen":"س","sheen":"ش","ta":"ط",
          "taa":"ت","thal":"ذ","thaa":"ث","toot":"ة","waw":"و","ya":"ي","zay":"ز"}
ar = lambda l: ARABIC.get(l, l)

WORDS_EASY = {
    "باب":["bb","aleff","bb"],"دار":["dal","aleff","ra"],"فاز":["fa","aleff","zay"],
    "زاد":["zay","aleff","dal"],"راس":["ra","aleff","seen"],"كلب":["kaaf","laam","bb"],
    "شال":["sheen","aleff","laam"],"قال":["gaaf","aleff","laam"],"جرس":["jeem","ra","seen"],
    "ثلج":["thaa","laam","jeem"],"فكر":["fa","kaaf","ra"],"لهب":["laam","ha","bb"],
    "ركب":["ra","kaaf","bb"],"كرز":["kaaf","ra","zay"],"بحر":["bb","haa","ra"],
    "نور":["nun","waw","ra"],"ذهب":["thal","ha","bb"],"ورد":["waw","ra","dal"],
    "شجر":["sheen","jeem","ra"],"قمر":["gaaf","meem","ra"],"نجم":["nun","jeem","meem"],
    "سمك":["seen","meem","kaaf"],"نهر":["nun","ha","ra"],
}
WORDS_HARD = {
    "كتاب":["kaaf","taa","aleff","bb"],"قلم":["gaaf","laam","meem"],"شمس":["sheen","meem","seen"],
    "مدرس":["meem","dal","ra","seen"],"بيت":["bb","ya","taa"],"قلب":["gaaf","laam","bb"],
    "كرسي":["kaaf","ra","seen","ya"],"قطار":["gaaf","ta","aleff","ra"],
    "نافذة":["nun","aleff","fa","thal","toot"],"مفتاح":["meem","fa","taa","aleff","haa"],
}
DIFFICULTY = {"Easy":WORDS_EASY,"Hard":WORDS_HARD,"All":{**WORDS_EASY,**WORDS_HARD}}

@st.cache_resource
def load_model():
    with open(CLASS_NAMES_JSON) as f: names = json.load(f)
    if len(names)!=31: st.error("❌ class_names.json: expected 31 classes"); return None,None,None
    ck=torch.load(MODEL_PATH,map_location=DEVICE)
    ck_n=ck.get("num_classes",None)
    if ck_n is not None and ck_n!=31: st.error(f"❌ Checkpoint has {ck_n} classes, expected 31."); return None,None,None
    state=ck.get("model_state_dict",ck)
    m=models.mobilenet_v2(weights=None); m.classifier[1]=nn.Linear(1280,31)
    try: m.load_state_dict(state)
    except RuntimeError as e: st.error(f"❌ {e}"); return None,None,None
    return m.to(DEVICE).eval(),names,31

transform = transforms.Compose([transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),
    transforms.ToTensor(),transforms.Normalize(config.NORMALIZATION_MEAN,config.NORMALIZATION_STD)])

def predict_image(pil_img, model, names):
    with torch.no_grad(): probs=F.softmax(model(transform(pil_img).unsqueeze(0).to(DEVICE)),dim=1)
    c,i=torch.topk(probs,k=min(3,probs.shape[1]),dim=1)
    return names[i[0][0].item()],c[0][0].item()*100,[(names[i[0][j].item()],c[0][j].item()*100) for j in range(i.shape[1])]

_font_cache,_text_cache={},{}
def get_font(size=28):
    if size not in _font_cache:
        for p in ["C:/Windows/Fonts/tahoma.ttf","C:/Windows/Fonts/arial.ttf"]:
            try: _font_cache[size]=ImageFont.truetype(p,size); break
            except OSError: pass
        if size not in _font_cache: _font_cache[size]=ImageFont.load_default()
    return _font_cache[size]

def draw_arabic_text(img, text, xy, font_size=28, color_bgr=(0,255,255)):
    key=(text,font_size,color_bgr)
    if key not in _text_cache:
        font=get_font(font_size); rgb=tuple(reversed(color_bgr))
        try:
            b=ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0),text,font=font)
            tw,th,ox,oy=b[2]-b[0]+6,b[3]-b[1]+6,-b[0]+3,-b[1]+3
        except AttributeError:
            tw,th=font.getsize(text); tw,th,ox,oy=tw+6,th+6,3,3
        patch=Image.new("RGBA",(tw,th),(0,0,0,0))
        ImageDraw.Draw(patch).text((ox,oy),text,font=font,fill=(*rgb,255))
        _text_cache[key]=np.array(patch)
    patch=_text_cache[key]; ph,pw=patch.shape[:2]; x,y=xy; fh,fw=img.shape[:2]
    x2,y2=min(x+pw,fw),min(y+ph,fh)
    if x2<=x or y2<=y: return img
    a=patch[:y2-y,:x2-x,3:4].astype(np.float32)/255.0
    t=patch[:y2-y,:x2-x,:3][:,:,::-1].astype(np.float32)
    out=img.copy(); out[y:y2,x:x2]=(img[y:y2,x:x2].astype(np.float32)*(1-a)+t*a).astype(np.uint8)
    return out

def _make_hand_detector():
    for tf in [APP_DIR/"src"/"app"/"hand_landmarker.task",APP_DIR/"hand_landmarker.task"]:
        if tf.exists():
            if not MEDIAPIPE_OK: return None
            opts=mp_vision.HandLandmarkerOptions(base_options=mp_tasks.BaseOptions(model_asset_path=str(tf)),
                num_hands=1,min_hand_detection_confidence=0.7,min_tracking_confidence=0.6)
            return mp_vision.HandLandmarker.create_from_options(opts)
    return None

def detect_hand_roi(frame, detector, pad=25):
    h,w=frame.shape[:2]
    res=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    if not res.hand_landmarks: return None,None,None
    lm=res.hand_landmarks[0]; xs=[p.x*w for p in lm]; ys=[p.y*h for p in lm]
    cx=(int(min(xs))+int(max(xs)))//2; cy=(int(min(ys))+int(max(ys)))//2
    half=max(int(max(xs))-int(min(xs)),int(max(ys))-int(min(ys)))//2+pad+5
    x1,y1=max(cx-half,0),max(cy-half,0); x2,y2=min(cx+half,w),min(cy+half,h)
    crop=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)[y1:y2,x1:x2]
    if crop.size==0: return None,None,None
    ch,cw=crop.shape[:2]
    if ch!=cw:
        t=max(ch,cw); crop=cv2.copyMakeBorder(crop,(t-ch)//2,t-ch-(t-ch)//2,(t-cw)//2,t-cw-(t-cw)//2,cv2.BORDER_REFLECT_101)
    return Image.fromarray(crop),(x1,y1,x2,y2),lm

def blur_background(frame, lm):
    h,w=frame.shape[:2]
    pts=np.array([[int(p.x*w),int(p.y*h)] for p in lm],dtype=np.int32)
    mask=np.zeros((h,w),dtype=np.uint8); cv2.fillConvexPoly(mask,cv2.convexHull(pts),255)
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    mf=cv2.GaussianBlur(mask.astype(np.float32),(21,21),0)/255.0
    return (frame*np.stack([mf]*3,axis=-1)+cv2.GaussianBlur(frame,(BLUR_K,BLUR_K),0)*(1-np.stack([mf]*3,axis=-1))).astype(np.uint8)

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.pred_buf=deque(maxlen=SMOOTHING); self.conf_buf=deque(maxlen=SMOOTHING)
        self.smoothed_prediction=self.model=self.class_names=None
        self.smoothed_confidence=0.0; self.hand_detected=False; self.top3=[]
        self.hands=_make_hand_detector(); self._n=0; self._last_bbox=self._last_lm=None
    def __del__(self):
        try: self.hands.close()
        except: pass
    def get_smoothed_prediction(self):
        if len(self.pred_buf)<MIN_VOTES: return None,0.0
        label,cnt=Counter(self.pred_buf).most_common(1)[0]
        if cnt<MIN_VOTES: return None,0.0
        confs=[c for p,c in zip(self.pred_buf,self.conf_buf) if p==label]
        return label,float(np.mean(confs)) if confs else 0.0
    def reset_buffers(self): self.pred_buf.clear(); self.conf_buf.clear()
    def recv(self, frame):
        img=frame.to_ndarray(format="bgr24")
        if self.model is None: return av.VideoFrame.from_ndarray(img,format="bgr24")
        self._n+=1
        if self._n%INFER_N==0: _,self._last_bbox,self._last_lm=detect_hand_roi(img,self.hands)
        bbox,lm=self._last_bbox,self._last_lm
        if bbox is not None:
            self.hand_detected=True; img=blur_background(img,lm)
            if self._n%INFER_N==0:
                try:
                    pred,conf,top3=predict_image(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),self.model,self.class_names)
                    self.top3=top3
                    if conf>=CONF_FRAME: self.pred_buf.append(pred); self.conf_buf.append(conf)
                except (RuntimeError,ValueError,TypeError) as e: print(e); self.reset_buffers()
            x1,y1,x2,y2=bbox; cv2.rectangle(img,(x1,y1),(x2,y2),(0,220,110),3)
        else: self.hand_detected=False; self.reset_buffers()
        self.smoothed_prediction,self.smoothed_confidence=self.get_smoothed_prediction()
        if self.smoothed_prediction and self.smoothed_confidence>=CONF_OK:
            lbl=f"{ar(self.smoothed_prediction)} ({self.smoothed_confidence:.1f}%)"; col=(0,220,110)
        elif self.smoothed_prediction: lbl=f"Uncertain ({self.smoothed_confidence:.1f}%)"; col=(100,160,255)
        else: lbl="Stabilizing..."; col=(180,180,180)
        return av.VideoFrame.from_ndarray(draw_arabic_text(img,lbl,(10,8),color_bgr=col),format="bgr24")

def init_state():
    ss=st.session_state
    for k,v in [("difficulty","Easy"),("selected_word",None),("current_index",0),("completed",False),
                ("score",0),("streak",0),("best_streak",0),("words_completed",0),
                ("total_correct",0),("total_wrong",0),("last_correct_time",0.0),
                ("last_wrong_time",0.0),("last_feedback",None),("auto_advance",True),
                ("balloons_shown",False),("word_start_time",time.time()),("best_time",None)]:
        if k not in ss: ss[k]=v
    if ss.selected_word not in DIFFICULTY[ss.difficulty]:
        ss.selected_word=random.choice(list(DIFFICULTY[ss.difficulty].keys()))

def reset_word():
    ss=st.session_state; ss.current_index=0; ss.completed=False; ss.last_feedback=None
    ss.balloons_shown=False; ss.last_correct_time=ss.last_wrong_time=0.0; ss.word_start_time=time.time()

def new_word():
    ss=st.session_state; ss.selected_word=random.choice(list(DIFFICULTY[ss.difficulty].keys())); reset_word()

def on_correct(letters, proc=None):
    ss=st.session_state; bonus=ss.streak*STREAK_B
    ss.score+=PTS+bonus; ss.streak+=1; ss.best_streak=max(ss.best_streak,ss.streak)
    ss.total_correct+=1; ss.current_index+=1; ss.last_correct_time=time.time()
    if ss.current_index>=len(letters):
        ss.completed=True; ss.words_completed+=1; t=time.time()-ss.word_start_time
        if ss.best_time is None or t<ss.best_time: ss.best_time=t
    ss.last_feedback=("success",f"✅ Correct! +{PTS}"+( f" (+{bonus} streak bonus)" if bonus else ""))
    if proc: proc.reset_buffers()

def on_wrong(pred, conf, target, proc=None):
    ss=st.session_state; ss.streak=0; ss.total_wrong+=1; ss.last_wrong_time=time.time()
    ss.last_feedback=("error",f"❌ Wrong. Got: **{ar(pred)}** ({conf:.1f}%) | Expected: **{ar(target)}**")
    if proc: proc.reset_buffers()

# ══════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ArSignTutor",page_icon="🤟",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#0a0e1a!important;color:#e8eaf6!important}
[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2a3a}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stMetric"]{background:linear-gradient(135deg,#111827,#1a2332);border:1px solid #2a3a4a;border-radius:12px;padding:12px 16px!important}
[data-testid="stMetricValue"]{font-size:28px!important;color:#00d4aa!important;font-weight:700}
[data-testid="stMetricLabel"]{color:#8899aa!important;font-size:12px!important}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1)!important;color:white!important;border:1px solid #1976d2!important;border-radius:10px!important;font-weight:600!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#00897b,#00695c)!important;border-color:#00acc1!important}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#00897b,#00d4aa)!important;border-radius:10px!important}
[data-testid="stProgress"]>div{background:#1a2332!important;border-radius:10px!important;height:10px!important}
hr{border-color:#1e2a3a!important}
</style>""",unsafe_allow_html=True)

if MODEL_PATH is None: st.error("❌ Model file not found."); st.stop()
if CLASS_NAMES_JSON is None: st.error("❌ class_names.json not found."); st.stop()
model,class_names,_=load_model()
if model is None: st.stop()

badges="".join(f"<span style='background:{b};border:1px solid {c};color:{t};padding:3px 12px;border-radius:20px;font-size:11px'>{lbl}</span>"
    for lbl,b,c,t in [("MobileNetV2","#0d3349","#1976d2","#64b5f6"),("MediaPipe","#0d3325","#00897b","#4db6ac"),("Acc 94.33%","#2d1b4e","#7b1fa2","#ce93d8"),("31 Letters","#1a2332","#37474f","#90a4ae")])
st.markdown(f"<div style='background:linear-gradient(135deg,#0d1b2a,#1a1035,#0d2137);border:1px solid #1e3a5f;border-radius:20px;padding:28px 40px;display:flex;align-items:center;gap:24px;box-shadow:0 8px 32px rgba(0,0,0,.5);margin-bottom:8px'><div style='font-size:68px'>🤟</div><div><div style='font-size:36px;font-weight:800;background:linear-gradient(90deg,#00d4aa,#4fc3f7,#ce93d8);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>ArSignTutor</div><div style='color:#8899aa;font-size:14px;margin:4px 0 10px'>Interactive Arabic Sign Language Learning System</div><div style='display:flex;gap:8px;flex-wrap:wrap'>{badges}</div></div></div>",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:12px 0;font-size:12px;color:#4fc3f7;letter-spacing:2px;font-weight:600'>STATISTICS</div>",unsafe_allow_html=True)
    init_state(); ss=st.session_state
    total=ss.total_correct+ss.total_wrong; acc=ss.total_correct/total*100 if total else 0.0
    c1,c2=st.columns(2); c1.metric("🏆 Score",ss.score); c2.metric("🔥 Streak",ss.streak)
    c3,c4=st.columns(2); c3.metric("⚡ Best",ss.best_streak); c4.metric("📝 Words",ss.words_completed)
    st.metric("🎯 Accuracy",f"{acc:.1f}%")
    if ss.best_time: st.metric("⏱️ Best Time",f"{ss.best_time:.1f}s")
    if total:
        st.markdown(f"<div style='background:#111827;border-radius:10px;padding:10px;margin:8px 0'><div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px'><span style='color:#4caf50'>✅ {ss.total_correct}</span><span style='color:#ef5350'>❌ {ss.total_wrong}</span></div><div style='background:#1a2332;border-radius:6px;height:8px'><div style='background:linear-gradient(90deg,#4caf50,#00d4aa);width:{acc:.1f}%;height:100%;border-radius:6px'></div></div></div>",unsafe_allow_html=True)
    st.divider()
    if st.button("🔄 Reset Stats",use_container_width=True):
        for k in ["score","streak","best_streak","words_completed","total_correct","total_wrong"]: ss[k]=0
        ss.best_time=None; st.rerun()
    st.divider()
    st.markdown("<div style='background:#0d1117;border:1px solid #1e2a3a;border-radius:12px;padding:12px;font-size:11px;color:#6a7f8f;line-height:2'>📊 <b style='color:#4fc3f7'>Model Info</b><br>• MobileNetV2 · PyTorch<br>• Acc: <span style='color:#00d4aa'>94.33%</span> · F1: <span style='color:#00d4aa'>94.36%</span><br>• 31 letters · no data leakage</div>",unsafe_allow_html=True)

col_diff,col_new=st.columns([3,1])
with col_diff:
    chosen=st.radio("",list(DIFFICULTY.keys()),index=list(DIFFICULTY.keys()).index(ss.difficulty),horizontal=True,label_visibility="collapsed")
    if chosen!=ss.difficulty: ss.difficulty=chosen; new_word(); st.rerun()
with col_new:
    if st.button("🔀 New Word",use_container_width=True,type="primary"): new_word(); st.rerun()

pool=DIFFICULTY[ss.difficulty]
if ss.selected_word not in pool: new_word(); st.rerun()
word=ss.selected_word; letters=pool[word]; idx=ss.current_index; elapsed=time.time()-ss.word_start_time

st.markdown(f"<div style='background:linear-gradient(135deg,#0d1b35,#1a1035);border:1px solid #2a3a5a;border-radius:18px;padding:24px 40px;text-align:center;margin:10px 0'><div style='font-size:11px;color:#4fc3f7;letter-spacing:3px;margin-bottom:8px'>SIGN THIS WORD LETTER BY LETTER</div><div style='font-size:80px;font-weight:900;background:linear-gradient(135deg,#fff,#b3d9ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:12px;direction:rtl'>{word}</div><div style='color:#4a6070;font-size:12px;margin-top:6px'>⏱️ {elapsed:.0f}s · {idx}/{len(letters)} letters</div></div>",unsafe_allow_html=True)

boxes="<div style='display:flex;gap:10px;flex-wrap:wrap;justify-content:center;direction:rtl;margin:8px 0'>"
for i,ltr in enumerate(letters):
    if i<idx:                         bg,bd,tc,badge="linear-gradient(135deg,#0a2e0a,#145214)","#27ae60","#69f0ae","✅"
    elif i==idx and not ss.completed: bg,bd,tc,badge="linear-gradient(135deg,#0d2b4e,#1565c0)","#4fc3f7","#fff","👇"
    else:                             bg,bd,tc,badge="linear-gradient(135deg,#111827,#1a2332)","#2a3a4a","#4a5568",""
    boxes+=f"<div style='text-align:center;padding:12px 16px;background:{bg};border:2px solid {bd};border-radius:14px;min-width:65px'><div style='font-size:42px;font-weight:bold;color:{tc}'>{ar(ltr)}</div><div style='font-size:10px;color:#4a5568'>{ltr}</div><div style='font-size:14px'>{badge}</div></div>"
boxes+="</div>"
st.markdown(boxes,unsafe_allow_html=True)
st.progress(idx/len(letters),text=f"Progress: {idx}/{len(letters)} letters")
st.divider()

if ss.completed:
    tt=time.time()-ss.word_start_time
    st.markdown(f"<div style='background:linear-gradient(135deg,#0a2e0a,#1b5e20);border:1px solid #43a047;border-radius:18px;padding:32px;text-align:center'><div style='font-size:48px'>🎉</div><div style='font-size:26px;font-weight:700;color:#69f0ae'>Well done! You completed «{word}» in {tt:.1f}s</div></div>",unsafe_allow_html=True)
    if not ss.balloons_shown: st.balloons(); ss.balloons_shown=True
    st.write("")
    if st.button("▶️ Start New Word",type="primary",use_container_width=True): new_word(); st.rerun()

else:
    target=letters[idx]; target_ar=ar(target); next_ar=ar(letters[idx+1]) if idx+1<len(letters) else ""
    parts=(f"<span style='font-size:38px;color:#69f0ae;font-weight:bold'>{''.join(ar(letters[i]) for i in range(idx))}</span>" if idx else "")+ \
          "<span style='font-size:38px;color:#4fc3f7;font-weight:bold;border-bottom:3px solid #4fc3f7;padding:0 6px;min-width:32px;display:inline-block;text-align:center'>_</span>"+ \
          "".join(f"<span style='font-size:38px;color:#2a3a4a;font-weight:bold'>{ar(letters[i])}</span>" for i in range(idx+1,len(letters)))
    st.markdown(f"<div style='direction:rtl;display:flex;align-items:center;gap:6px;margin-bottom:14px;flex-wrap:wrap;justify-content:center'>{parts}</div>",unsafe_allow_html=True)

    col_card,col_cam,col_status=st.columns([1,2,1])
    with col_card:
        st.markdown(f"<div style='background:linear-gradient(160deg,#0d2b4e,#1a3a6e);border:1px solid #2a5a8a;border-radius:18px;padding:24px 14px;text-align:center;color:white'><div style='font-size:11px;color:#90caf9;letter-spacing:2px;margin-bottom:10px'>SIGN THIS LETTER</div><div style='font-size:110px;font-weight:900;line-height:1;background:linear-gradient(135deg,#fff,#90caf9);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>{target_ar}</div><div style='font-size:12px;color:#4a7a9a;margin-top:6px'>({target})</div></div>",unsafe_allow_html=True)
        if next_ar:
            st.markdown(f"<div style='text-align:center;padding:10px;background:#0d1117;border:1px solid #1e2a3a;border-radius:12px;margin-top:8px'><div style='font-size:10px;color:#4a5568;letter-spacing:2px'>NEXT</div><div style='font-size:34px;color:#6a7f8f;font-weight:bold'>{next_ar}</div></div>",unsafe_allow_html=True)

    with col_cam:
        ctx=webrtc_streamer(key="tutor",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIG,
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video":{"width":{"ideal":480},"height":{"ideal":360}},"audio":False},
            async_processing=True)
        if ctx.video_processor:
            ctx.video_processor.model=model; ctx.video_processor.class_names=class_names

    with col_status:
        st.markdown("<div style='font-size:12px;color:#4fc3f7;letter-spacing:2px;font-weight:600;margin-bottom:8px'>RECOGNITION STATUS</div>",unsafe_allow_html=True)
        if ctx.video_processor:
            vp=ctx.video_processor
            hbg,hbd,htc=("#0a2e1a","#27ae60","#69f0ae") if vp.hand_detected else ("#2e0a0a","#c0392b","#ef9a9a")
            st.markdown(f"<div style='background:{hbg};border:1px solid {hbd};border-radius:10px;padding:8px;text-align:center;color:{htc};font-size:12px;margin-bottom:8px'>{'✋ Hand detected' if vp.hand_detected else '❌ No hand'}</div>",unsafe_allow_html=True)
            if vp.smoothed_prediction:
                conf=vp.smoothed_confidence; confident=conf>=CONF_OK; match=vp.smoothed_prediction==target
                bg_c,bd_c,tc=("linear-gradient(135deg,#0a2e0a,#145214)","#27ae60","#69f0ae") if (match and confident) else \
                              ("linear-gradient(135deg,#2e0a0a,#5c1414)","#c0392b","#ef9a9a") if confident else \
                              ("linear-gradient(135deg,#1a1a0a,#2e2a0a)","#f39c12","#ffe082")
                label_txt=ar(vp.smoothed_prediction) if confident else "Uncertain"
                icon="✅" if (match and confident) else ("❌" if confident else "⚠️")
                st.markdown(f"<div style='text-align:center;padding:14px;background:{bg_c};border:2px solid {bd_c};border-radius:14px;color:white'><div style='font-size:11px;color:#889;margin-bottom:4px'>Reading:</div><div style='font-size:52px;font-weight:900;color:{tc}'>{label_txt}</div><div style='font-size:12px;color:#8899aa'>{icon} {conf:.1f}%</div></div>",unsafe_allow_html=True)
                for rl,rc in vp.top3: st.caption(f"{'🟢' if rl==target else '⚪'} {ar(rl)} — {rc:.1f}%")
            else:
                st.markdown("<div style='text-align:center;padding:18px;background:#111827;border:1px solid #2a3a4a;border-radius:14px;color:#4a5568'><div style='font-size:24px'>⏳</div><div style='font-size:11px;margin-top:4px'>Stabilizing...</div></div>",unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center;padding:24px;background:#111827;border:1px solid #2a3a4a;border-radius:14px;color:#4a5568'><div style='font-size:28px'>📷</div><div style='font-size:11px;margin-top:6px'>Click START</div></div>",unsafe_allow_html=True)

    if ss.last_feedback:
        kind,msg=ss.last_feedback
        bg_f,bd_f,tc_f=("#0a2e0a","#27ae60","#69f0ae") if kind=="success" else ("#2e0a0a","#c0392b","#ef9a9a")
        st.markdown(f"<div style='background:{bg_f};border:1px solid {bd_f};border-radius:10px;padding:10px 14px;color:{tc_f};direction:rtl'>{msg}</div>",unsafe_allow_html=True)

    now=time.time(); cd_ok=now-ss.last_correct_time<CD_OK; cd_wr=now-ss.last_wrong_time<CD_WRONG
    if ss.auto_advance and ctx.video_processor and not(cd_ok or cd_wr):
        vp=ctx.video_processor
        if vp.smoothed_prediction and vp.smoothed_confidence>=CONF_OK:
            if vp.smoothed_prediction==target: on_correct(letters,vp)
            else: on_wrong(vp.smoothed_prediction,vp.smoothed_confidence,target,vp)
            st.rerun()
    if cd_ok:
        st.markdown(f"<div style='background:#0d2b4e;border:1px solid #1976d2;border-radius:10px;padding:10px;color:#90caf9'>⏳ Get ready for next letter... ({CD_OK-(now-ss.last_correct_time):.1f}s)</div>",unsafe_allow_html=True)
    elif cd_wr:
        st.markdown(f"<div style='background:#2e1a0a;border:1px solid #f39c12;border-radius:10px;padding:10px;color:#ffe082'>⚠️ Try again... ({CD_WRONG-(now-ss.last_wrong_time):.1f}s)</div>",unsafe_allow_html=True)

    st.write("")
    col_check,col_skip,col_toggle=st.columns(3)
    with col_check:
        if st.button("✋ Check Manually",use_container_width=True):
            if not ctx.video_processor: st.warning("Start the camera first")
            else:
                vp=ctx.video_processor
                if not vp.smoothed_prediction: st.warning("Still stabilizing...")
                elif vp.smoothed_confidence<CONF_OK: st.warning(f"Low confidence ({vp.smoothed_confidence:.1f}%)")
                elif vp.smoothed_prediction==target: on_correct(letters,vp); st.rerun()
                else: on_wrong(vp.smoothed_prediction,vp.smoothed_confidence,target,vp); st.rerun()
    with col_skip:
        if st.button("⏭️ Skip",use_container_width=True):
            ss.current_index+=1; ss.streak=0; ss.last_correct_time=time.time()
            ss.last_feedback=("error",f"⏭️ Skipped {ar(target)}")
            if ss.current_index>=len(letters): ss.completed=True; ss.words_completed+=1
            st.rerun()
    with col_toggle: ss.auto_advance=st.toggle("Auto Advance",value=ss.auto_advance)
    if ss.auto_advance and ctx.state.playing and ctx.video_processor: time.sleep(0.15); st.rerun()

st.divider()
st.markdown("<div style='display:flex;justify-content:space-between;color:#3a4f5f;font-size:11px;flex-wrap:wrap;gap:8px'>🤟 ArSignTutor · Arabic Sign Language<span>MobileNetV2 · Accuracy 94.33%</span><span>AI Department © 2026</span></div>",unsafe_allow_html=True)
