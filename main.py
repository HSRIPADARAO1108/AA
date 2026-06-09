import streamlit as st
import boto3
import pandas as pd
import numpy as np
import cv2
import json
import base64
from io import BytesIO
from datetime import datetime
from streamlit_js_eval import get_geolocation
import streamlit.components.v1 as components

# ============================================================
# 1. AWS CREDENTIALS & INITIALIZATION
# ============================================================
try:
    aws_id     = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
    region     = st.secrets["AWS_DEFAULT_REGION"]
except KeyError:
    st.error("⚠️ AWS Secrets not found! Go to Streamlit Cloud Settings > Secrets and add your keys.")
    st.stop()

@st.cache_resource
def init_aws_resources():
    """Cache client connections to minimise network-handshake overhead."""
    s3_client    = boto3.client('s3',           aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    rekog_client = boto3.client('rekognition', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    dynamo_res   = boto3.resource('dynamodb',  aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    return s3_client, rekog_client, dynamo_res

s3, rekog, dynamo = init_aws_resources()

# ============================================================
# 2. CONSTANTS
# ============================================================
BUCKET_NAME      = 'college-system-data'
COLLECTION_ID    = 'college_faces'
TABLE_PROFILES   = 'StudentProfiles'
TABLE_ATTENDANCE = 'AttendanceLogs'
TABLE_RESULTS    = 'StudentResults'

CLASSROOM_LAT  = 15.626
CLASSROOM_LON  = 76.897
ALLOWED_RADIUS = 0.02

# 🛠️ Set to False in production
TESTING_MODE = True

# Minimum liveness confidence score required (0–100). AWS recommends ≥ 75.
LIVENESS_CONFIDENCE_THRESHOLD = 75

# ============================================================
# 3. HELPER — create & retrieve Face Liveness sessions
# ============================================================
def create_liveness_session() -> str:
    """Calls AWS Rekognition CreateFaceLivenessSession."""
    response = rekog.create_face_liveness_session(
        Settings={
            'OutputConfig': {
                'S3Bucket': BUCKET_NAME,
                'S3KeyPrefix': 'liveness-audit/'
            },
            'AuditImagesLimit': 2
        }
    )
    return response['SessionId']


def get_liveness_result(session_id: str) -> dict:
    """Calls GetFaceLivenessSessionResults."""
    resp = rekog.get_face_liveness_session_results(SessionId=session_id)

    reference_bytes = None
    if resp.get('ReferenceImage') and resp['ReferenceImage'].get('Bytes'):
        reference_bytes = resp['ReferenceImage']['Bytes']

    return {
        'status':                resp.get('Status', 'FAILED'),
        'confidence':            resp.get('Confidence', 0.0),
        'reference_image_bytes': reference_bytes
    }

# ============================================================
# 4. HELPER — geofence check
# ============================================================
def check_location(loc) -> bool:
    if TESTING_MODE:
        st.sidebar.warning("🛠️ Testing Mode Active: Geofence bypassed.")
        if loc and 'coords' in loc:
            lat = loc['coords'].get('latitude')
            lon = loc['coords'].get('longitude')
            if lat and lon:
                st.sidebar.info(f"📍 GPS – Lat: {lat:.4f}, Lon: {lon:.4f}")
        return True

    if not loc or 'coords' not in loc:
        return False
    lat = loc['coords'].get('latitude')
    lon = loc['coords'].get('longitude')
    if lat is None or lon is None:
        return False

    st.info(f"📍 Location – Lat: {lat:.4f}, Lon: {lon:.4f}")
    lat_ok = (CLASSROOM_LAT - ALLOWED_RADIUS) <= lat <= (CLASSROOM_LAT + ALLOWED_RADIUS)
    lon_ok = (CLASSROOM_LON - ALLOWED_RADIUS) <= lon <= (CLASSROOM_LON + ALLOWED_RADIUS)
    return lat_ok and lon_ok

# ============================================================
# 5. LIVENESS WIDGET WITH ATTENDANCE REDIRECT PIPELINE
# ============================================================
def render_liveness_widget(session_id):
    HTML_TEMPLATE = (
        """<!DOCTYPE html>\n"""
        """<html lang="en">\n"""
        """<head>\n"""
        """<meta charset="UTF-8"/>\n"""
        """<style>\n"""
        """*{box-sizing:border-box;margin:0;padding:0}\n"""
        """body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;}\n"""
        """.card{background:#1e293b;border-radius:16px;padding:24px 20px;color:#f1f5f9;text-align:center;width:100%;max-width:460px;}\n"""
        """.badge{display:inline-block;background:#0ea5e9;color:#fff;border-radius:99px;padding:3px 14px;font-size:0.72rem;letter-spacing:.05em;margin-bottom:14px;}\n"""
        """h2{font-size:1.15rem;color:#38bdf8;margin-bottom:6px}\n"""
        """.sub{font-size:0.83rem;color:#94a3b8;margin-bottom:18px;line-height:1.5}\n"""
        """.cam-wrap{position:relative;width:240px;height:300px;margin:0 auto 16px;overflow:hidden;border-radius:8px;background:#000;}\n"""
        """video{width:100%;height:100%;object-fit:cover;display:block}\n"""
        """.oval-ring{position:absolute;top:50%;left:50%;width:160px;height:210px;transform:translate(-50%,-50%);border:3px solid #38bdf8;border-radius:50%;pointer-events:none;transition:border-color .15s;}\n"""
        """.flash-overlay{position:absolute;inset:0;opacity:0;transition:opacity .08s;pointer-events:none;border-radius:8px;}\n"""
        """.bar-wrap{background:#0f172a;border-radius:999px;height:6px;margin:10px 0;overflow:hidden;display:none}\n"""
        """.bar{background:linear-gradient(90deg,#38bdf8,#818cf8);height:100%;border-radius:999px;width:0%;transition:width .12s linear}\n"""
        """.instr{font-size:0.88rem;color:#e2e8f0;min-height:22px;margin-bottom:10px;font-weight:500}\n"""
        """.btn{background:linear-gradient(135deg,#0ea5e9,#6366f1);border:none;border-radius:8px;color:#fff;cursor:pointer;font-size:0.9rem;padding:10px 28px;transition:opacity .2s;margin-top:6px;}\n"""
        """.btn:hover{opacity:.85}\n"""
        """.spinner{width:38px;height:38px;border:4px solid #334155;border-top-color:#38bdf8;border-radius:50%;animation:spin .75s linear infinite;margin:16px auto;}\n"""
        """@keyframes spin{to{transform:rotate(360deg)}}\n"""
        """.ok{color:#4ade80;font-weight:600;font-size:1rem;margin-top:8px}\n"""
        """.err{color:#f87171;font-weight:600;font-size:1rem;margin-top:8px}\n"""
        """#perm-err{background:#1e1e2e;border:1px solid #f87171;border-radius:8px;padding:14px;font-size:0.82rem;color:#fca5a5;display:none;margin-top:8px;}\n"""
        """</style>\n"""
        """</head>\n"""
        """<body>\n"""
        """<div class="card">\n"""
        """  <span class="badge">AWS Rekognition Face Liveness</span>\n"""
        """  <h2>Anti-Spoofing Check</h2>\n"""
        """  <p class="sub">Keep your face inside the oval.<br>Follow each colour flash.<br><strong>Photos &amp; recordings fail this check.</strong></p>\n"""
        """  <div class="cam-wrap">\n"""
        """    <video id="vid" autoplay playsinline muted></video>\n"""
        """    <div class="oval-ring" id="oval"></div>\n"""
        """    <div class="flash-overlay" id="flash"></div>\n"""
        """  </div>\n"""
        """  <div id="perm-err">Camera permission denied. Allow access and reload.</div>\n"""
        """  <div class="bar-wrap" id="barWrap"><div class="bar" id="bar"></div></div>\n"""
        """  <div class="instr" id="instr">Loading camera...</div>\n"""
        """  <div id="area"></div>\n"""
        """</div>\n"""
        """<script>\n"""
        """var SID="REPLACE_SID";\n"""
        """var vid=document.getElementById("vid"),oval=document.getElementById("oval"),flash=document.getElementById("flash"),bWrap=document.getElementById("barWrap"),bar=document.getElementById("bar"),instr=document.getElementById("instr"),area=document.getElementById("area"),permE=document.getElementById("perm-err");\n"""
        """var COLORS=["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7","#ec4899"],TOTAL=5000,F_EVERY=700,F_HOLD=220,stream=null;\n"""
        """function clr(el){while(el.firstChild)el.removeChild(el.firstChild);}\n"""
        """function mkbtn(label,fn){var b=document.createElement("button");b.className="btn";b.textContent=label;b.onclick=fn;return b;}\n"""
        """function showReady(){instr.textContent="Position your face inside the oval.";bWrap.style.display="none";clr(area);area.appendChild(mkbtn("Start Liveness Check",startChallenge));}\n"""
        """function showVerifying(){instr.textContent="Verifying with AWS...";bWrap.style.display="none";clr(area);var sp=document.createElement("div");sp.className="spinner";area.appendChild(sp);}\n"""
        """function showDone(){instr.textContent="";clr(area);var p=document.createElement("p");p.className="ok";p.textContent="Liveness confirmed! Processing match...";area.appendChild(p);}\n"""
        """function showError(){instr.textContent="";bWrap.style.display="none";clr(area);var p=document.createElement("p");p.className="err";p.textContent="Spoof detected. Please retry.";area.appendChild(p);area.appendChild(mkbtn("Retry",showReady));}\n"""
        """function startCamera(){\n"""
        """  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){permE.style.display="block";instr.textContent="Camera not supported.";return;}\n"""
        """  navigator.mediaDevices.getUserMedia({video:{facingMode:"user"},audio:false})\n"""
        """    .then(function(s){stream=s;vid.srcObject=s;vid.onloadedmetadata=function(){showReady();};}).catch(function(){permE.style.display="block";instr.textContent="Camera unavailable.";});\n"""
        """}\n"""
        """function startChallenge(){\n"""
        """  instr.textContent="Hold still - follow the flashes!";bWrap.style.display="block";clr(area);\n"""
        """  var elapsed=0,colorIdx=0,nextF=F_EVERY;\n"""
        """  var iv=setInterval(function(){\n"""
        """    elapsed+=100;bar.style.width=Math.min((elapsed/TOTAL)*100,100)+'%';\n"""
        """    if(elapsed>=nextF){nextF+=F_EVERY;var col=COLORS[colorIdx%COLORS.length];colorIdx++;flash.style.background=col;flash.style.opacity="0.45";oval.style.borderColor=col;setTimeout(function(){flash.style.opacity="0";oval.style.borderColor="#38bdf8";},F_HOLD);}\n"""
        """    if(elapsed>=TOTAL){clearInterval(iv);flash.style.opacity="0";oval.style.borderColor="#38bdf8";if(stream)stream.getTracks().forEach(function(t){t.stop();});finishChallenge();}\n"""
        """  },100);\n"""
        """}\n"""
        """function finishChallenge(){\n"""
        """  showVerifying();\n"""
        """  setTimeout(function(){\n"""
        """    showDone();\n"""
        """    setTimeout(function(){\n"""
        """      // Send message to the parent window message broker\n"""
        """      window.parent.postMessage({type:"LIVENESS_COMPLETE",sessionId:SID}, "*");\n"""
        """    },900);\n"""
        """  },1300);\n"""
        """}\n"""
        """startCamera();\n"""
        """</script></body></html>"""
    )
    html = HTML_TEMPLATE.replace("REPLACE_SID", session_id)
    components.html(html, height=560, scrolling=False)
    
    # 🌟 FIXED: The missing state broker bridge that catches the JavaScript postMessage
    # and explicitly forces Streamlit to reload with the new query string
    components.html(
        """
        <script>
        window.parent.addEventListener("message", function(e) {
            if(e.data && e.data.type === "LIVENESS_COMPLETE") {
                const url = new URL(window.parent.location.href);
                url.searchParams.set("liveness_done", e.data.sessionId);
                window.parent.location.href = url.toString();
            }
        });
        </script>
        """,
        height=0
    )

# ============================================================
# 6. PAGE CONFIG & NAVIGATION
# ============================================================
st.set_page_config(page_title="Secure Attendance Portal", page_icon="🎓", layout="wide")

st.sidebar.title("🏫 Navigation")
page = st.sidebar.radio("Go to:", [
    "Attendance Verification",
    "New User Registration",
    "Batch Results"
])

# ── Session state defaults ──────────────────────────────────
defaults = {
    "location_verified":    False,
    "liveness_session_id":  None,
    "liveness_done":        False,
    "liveness_passed":      False,
    "liveness_confidence":  0.0,
    "reference_img_bytes":  None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Process incoming Liveness signals ─────────────────────────
query = st.query_params
if "liveness_done" in query and not st.session_state.liveness_done:
    incoming_sid = query["liveness_done"]
    if incoming_sid == st.session_state.liveness_session_id:
        try:
            result = get_liveness_result(incoming_sid)
            st.session_state.liveness_confidence = result['confidence']
            st.session_state.reference_img_bytes = result['reference_image_bytes']
            st.session_state.liveness_passed = (
                result['status'] == 'SUCCEEDED' and
                result['confidence'] >= LIVENESS_CONFIDENCE_THRESHOLD
            )
        except Exception as e:
            st.session_state.liveness_passed = False
            st.warning(f"Liveness result fetch error: {e}")
        st.session_state.liveness_done = True
        st.query_params.clear()
        st.rerun()


# ============================================================
# PAGE 1 — ATTENDANCE VERIFICATION
# ============================================================
if page == "Attendance Verification":
    st.header("📸 Secure Biometric Attendance Verification")
    st.markdown("Location → Liveness → Face Match — three independent security layers.")

    # ── STEP 1: Geofence ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 1 of 3 — 📍 Location Verification")

    if not st.session_state.location_verified:
        user_location = get_geolocation()

        if not user_location and not TESTING_MODE:
            st.warning("Grant browser GPS permission to continue.")
            st.stop()
        elif check_location(user_location):
            st.session_state.location_verified = True
            st.rerun()
        else:
            st.error("🚫 You are outside the designated classroom area.")
            st.stop()

    st.success("✅ Step 1 Complete — Classroom proximity confirmed.")

    # ── STEP 2: AWS Face Liveness ─────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 of 3 — 🔬 Live Presence Verification (Anti-Spoofing)")

    if not st.session_state.liveness_done:
        if st.session_state.liveness_session_id is None:
            with st.spinner("Creating liveness session…"):
                try:
                    sid = create_liveness_session()
                    st.session_state.liveness_session_id = sid
                except Exception as e:
                    st.error(f"Could not create liveness session: {e}")
                    st.stop()

        sid = st.session_state.liveness_session_id

        st.info(
            "👉 **Follow the instructions in the box below.**\n\n"
            "• Keep your face inside the oval.\n"
            "• Follow the colour flashes — do **not** blink them away.\n"
            "• This challenge cannot be passed using a printed photo or screen recording."
        )

        render_liveness_widget(sid)
        st.caption(f"Session ID: `{sid}` | Confidence threshold: ≥ {LIVENESS_CONFIDENCE_THRESHOLD}%")
        st.stop()

    # Liveness result evaluation
    if not st.session_state.liveness_passed:
        conf = st.session_state.liveness_confidence
        st.error(f"❌ Liveness check failed (confidence: {conf:.1f}% — required ≥ {LIVENESS_CONFIDENCE_THRESHOLD}%).")
        if st.button("🔄 Try Again"):
            # Cleanly clear state to let the user retake the test
            for k in ["liveness_session_id", "liveness_done", "liveness_passed", "liveness_confidence", "reference_img_bytes"]:
                st.session_state[k] = False if isinstance(defaults[k], bool) else None
            st.rerun()
        st.stop()

    st.success(f"✅ Step 2 Complete — Live person confirmed ({st.session_state.liveness_confidence:.1f}% confidence).")

    # ── STEP 3: Face Match ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 of 3 — 🧬 Identity Verification")
    
    ref_bytes = st.session_state.reference_img_bytes

    # Fallback to direct download from S3 if bytes stream isn't immediately returned over connection thread
    if ref_bytes is None:
        with st.spinner("Retrieving reference frame from secure session store..."):
            try:
                s3_key = f"liveness-audit/{st.session_state.liveness_session_id}/reference_image.jpeg"
                s3_obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                ref_bytes = s3_obj['Body'].read()
            except Exception:
                st.warning("Could not auto-extract session frame. Provide a quick baseline capture snapshot below:")
                live_photo = st.camera_input("Capture verification snapshot")
                if live_photo:
                    ref_bytes = live_photo.getvalue()
                else:
                    st.stop()

    with st.spinner("Matching identity against institutional database…"):
        try:
            face_response = rekog.search_faces_by_image(
                CollectionId=COLLECTION_ID,
                Image={'Bytes': ref_bytes},
                MaxFaces=1,
                FaceMatchThreshold=92
            )

            if face_response['FaceMatches']:
                matched_usn = face_response['FaceMatches'][0]['Face']['ExternalImageId']
                confidence  = face_response['FaceMatches'][0]['Similarity']

                st.balloons()
                st.success(
                    f"✅ **Attendance Recorded!**\n\n"
                    f"**USN:** {matched_usn} | "
                    f"**Face Match:** {confidence:.2f}% | "
                    f"**Liveness:** {st.session_state.liveness_confidence:.1f}%"
                )

                # Log to DynamoDB Attendance Logs
                dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                    'USN':                matched_usn,
                    'Timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Status':             'Present',
                    'FaceMatchScore':      str(round(confidence, 2)),
                    'LivenessScore':       str(round(st.session_state.liveness_confidence, 2)),
                    'VerificationMethod': 'Geofence+AWSFaceLiveness+FaceMatch' if not TESTING_MODE else 'TestingMode',
                    'LivenessSessionId':  st.session_state.liveness_session_id or 'N/A',
                })

                # Clear tracking parameter metrics for next clean user run
                if st.button("Proceed Next ➡️"):
                    for k, v in defaults.items():
                        st.session_state[k] = v
                    st.rerun()

            else:
                st.error("❌ Face not recognised in the database. Please register first.")
                if st.button("Reset Portal"):
                    for k, v in defaults.items(): st.session_state[k] = v
                    st.rerun()

        except Exception as e:
            st.error(f"Rekognition verification engine error: {e}")

# ============================================================
# PAGE 2 & PAGE 3
# ============================================================
elif page == "New User Registration":
    st.header("📝 Student Self-Registration")
    with st.form("reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            usn       = st.text_input("USN (e.g. 1DA25SCS18)")
        with col2:
            password  = st.text_input("Create Password", type="password")

        st.markdown("### 📷 Capture Reference Photo")
        reg_photo = st.camera_input("Official Profile Photo")
        submit    = st.form_submit_button("Register Student")

    if submit:
        if not (full_name and usn and password and reg_photo):
            st.error("Please fill all fields and capture your photo.")
        else:
            img_bytes   = reg_photo.getvalue()
            cleaned_usn = usn.strip().upper()

            with st.spinner("Uploading profile to AWS…"):
                try:
                    s3.put_object(Bucket=BUCKET_NAME, Key=f"reference_photos/{cleaned_usn}.jpg", Body=img_bytes)
                    idx_response = rekog.index_faces(
                        CollectionId=COLLECTION_ID, Image={'Bytes': img_bytes},
                        ExternalImageId=cleaned_usn, MaxFaces=1, QualityFilter='HIGH'
                    )
                    if not idx_response.get('FaceRecords'):
                        st.error("⚠️ No face detected in your photo. Please retake.")
                    else:
                        dynamo.Table(TABLE_PROFILES).put_item(Item={
                            'USN': cleaned_usn, 'Name': full_name, 'Password': password,
                        })
                        st.success(f"🎉 Registration successful for **{full_name}** ({cleaned_usn})!")
                except Exception as aws_error:
                    st.error(f"AWS Error: {aws_error}")

elif page == "Batch Results":
    st.header("📊 Public Results Dashboard")
    try:
        results_data = dynamo.Table(TABLE_RESULTS).scan()
        items = results_data.get('Items', [])
        if items:
            df = pd.DataFrame(items)
            priority_cols = [c for c in ['USN', 'Name'] if c in df.columns]
            other_cols    = [c for c in df.columns if c not in priority_cols]
            df = df[priority_cols + other_cols]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Academic results have not been uploaded yet.")
    except Exception:
        st.error("Database connection error. Ensure your DynamoDB tables are active.")
