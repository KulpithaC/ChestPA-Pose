import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from pose_checker import analyze_pose
import av
from PIL import Image, ImageDraw, ImageFont

# Configure MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Real-time Chest PA Pose Checker")
st.title("📹 ตรวจจับท่า Chest PA แบบเรียลไทม์")

# WebRTC configuration for deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

THAI_FONT_PATH = "fonts/THSarabun.ttf" 

def draw_text_pil(img, text, position, color=(0,255,0), font_size=32):
    """Draw Thai text on OpenCV image using PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(THAI_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.feedback_messages = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontally to remove mirror effect
        img = cv2.flip(img, 1)
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Get feedback
            feedback = analyze_pose(results.pose_landmarks.landmark)
            self.feedback_messages = feedback
            
            # Draw feedback text on image
            y = 30
            for msg in feedback:
                color = (0, 255, 0) if "✅" in msg else (0, 0, 255)
                img = draw_text_pil(img, msg, (10, y), color=color, font_size=32)
                y += 40
        else:
            self.feedback_messages = ["❌ ไม่พบการตรวจจับท่าทาง"]
            img = draw_text_pil(img, "ไม่พบการตรวจจับท่าทาง", (10, 30), color=(0, 0, 255), font_size=32)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Instructions
st.markdown("""
### 📋 วิธีใช้งาน:
1. คลิก "Start" เพื่อเปิดกล้อง
2. อนุญาตการเข้าถึงกล้องเมื่อเบราว์เซอร์ถาม
3. วางตัวให้อยู่ในกรอบกล้อง
4. ดูผลการวิเคราะห์ท่าทางแบบเรียลไทม์

### 🎯 เกณฑ์การตรวจสอบ:
- ✅ ไหล่เท่ากัน
- ✅ ศีรษะอยู่ตรงกลาง  
- ✅ วางมือบริเวณสะโพก
- ✅ โน้มไหล่ไปด้านหน้า
""")

col1, col2 = st.columns([2, 1])

with col1:
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="pose-detection",
        video_processor_factory=PoseVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# with col2:
    
#     if webrtc_ctx.video_transformer:
#         feedback_placeholder = st.empty()
        
#         # Display feedback in real-time
#         if hasattr(webrtc_ctx.video_transformer, 'feedback_messages'):
#             feedback = webrtc_ctx.video_transformer.feedback_messages
            
#             for msg in feedback:
#                 if "✅" in msg:
#                     feedback_placeholder.success(msg)
#                 elif "❌" in msg:
#                     feedback_placeholder.error(msg)
#                 else:
#                     feedback_placeholder.info(msg)

# Troubleshooting section
st.markdown("""
---
### 🔧 หากมีปัญหา:
- **กล้องไม่เปิด**: ตรวจสอบว่าอนุญาตการเข้าถึงกล้องแล้ว
- **ช้า**: ลองรีเฟรชหน้าเว็บ
- **ไม่ตรวจจับท่า**: ตรวจสอบแสงและตำแหน่งในกรอบกล้อง
""")

# Add CSS for better styling
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.main-header {
    text-align: center;
    color: #1e3a8a;
    font-size: 2rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)