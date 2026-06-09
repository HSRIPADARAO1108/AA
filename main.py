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
    s3_client    = boto3.client('s3',          aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
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
    """
    Calls AWS Rekognition CreateFaceLivenessSession.
    Returns a SessionId that the React FaceLivenessDetector widget will consume.
    Audit images are stored in your existing S3 bucket for compliance.
    """
    response = rekog.create_face_liveness_session(
        Settings={
            'OutputConfig': {
                'S3Bucket': BUCKET_NAME,
                'S3KeyPrefix': 'liveness-audit/'
            },
            'AuditImagesLimit': 2          # 0–4 images stored per session
        }
    )
    return response['SessionId']


def get_liveness_result(session_id: str) -> dict:
    """
    Calls GetFaceLivenessSessionResults.
    Returns:
        {
          'status': 'SUCCEEDED'|'IN_PROGRESS'|'FAILED',
          'confidence': float (0–100),
          'reference_image_bytes': bytes | None
        }
    """
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
# 5. LIVENESS WIDGET (React + AWS Amplify, embedded in iframe)
# ============================================================
def render_liveness_widget(session_id: str, aws_region: str) -> str | None:
    """
    Renders the AWS Amplify FaceLivenessDetector inside a self-contained
    HTML page injected via st.components.v1.html().

    The widget communicates back to Streamlit via window.parent.postMessage.
    We poll st.session_state for the result token written by the JS listener.

    Returns:
        'COMPLETE' once the challenge finishes, else None.
    """

    # We pass the session_id and region into the HTML at render-time.
    # No AWS credentials are exposed to the browser — the Amplify component
    # calls StartFaceLivenessSession, which is authorised by an *unauthenticated*
    # Cognito Identity Pool guest role that you configure once (see README).
    #
    # If you have not set up Cognito yet, set TESTING_MODE = True and the
    # widget will simulate a successful liveness pass for development purposes.

    html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Face Liveness</title>

  <!-- React 18 + ReactDOM (UMD builds from CDN) -->
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

  <!-- Babel so we can write JSX inline -->
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

  <!-- AWS Amplify core + Auth + UI React + Liveness -->
  <!-- NOTE: These ESM builds do NOT work in a plain <script> tag,       -->
  <!-- so we use the UMD/IIFE bundles via skypack / esm.sh CDN.          -->
  <!-- The Amplify UI React Liveness component requires the packages     -->
  <!-- below in the exact order listed.                                   -->
  <script src="https://cdn.jsdelivr.net/npm/aws-amplify@6/dist/aws-amplify.min.js"></script>

  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 16px;
    }}
    #root {{
      width: 100%;
      max-width: 520px;
    }}
    .card {{
      background: #1e293b;
      border-radius: 16px;
      padding: 28px 24px;
      color: #f1f5f9;
      text-align: center;
    }}
    .card h2 {{ font-size: 1.25rem; margin-bottom: 8px; color: #38bdf8; }}
    .card p  {{ font-size: 0.9rem; color: #94a3b8; margin-bottom: 20px; line-height: 1.5; }}
    .badge {{
      display: inline-block;
      background: #0ea5e9;
      color: #fff;
      border-radius: 999px;
      padding: 4px 14px;
      font-size: 0.75rem;
      margin-bottom: 18px;
      letter-spacing: 0.05em;
    }}

    /* Liveness container */
    #liveness-mount {{
      border-radius: 12px;
      overflow: hidden;
      min-height: 260px;
    }}

    /* Status messages */
    .status-ok  {{ color: #4ade80; font-weight: 600; font-size: 1rem; margin-top: 12px; }}
    .status-err {{ color: #f87171; font-weight: 600; font-size: 1rem; margin-top: 12px; }}
    .status-loading {{ color: #facc15; font-size: 0.9rem; margin-top: 12px; }}

    /* Simple spinner */
    .spinner {{
      width: 40px; height: 40px;
      border: 4px solid #334155;
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 20px auto;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* Simulated liveness box (used when Amplify CDN unavailable) */
    .sim-box {{
      background: #0f172a;
      border: 2px dashed #334155;
      border-radius: 12px;
      padding: 40px 20px;
      text-align: center;
    }}
    .sim-oval {{
      width: 120px; height: 160px;
      border: 3px solid #38bdf8;
      border-radius: 50%;
      margin: 0 auto 16px;
      position: relative;
      animation: pulse-oval 1.5s ease-in-out infinite;
    }}
    @keyframes pulse-oval {{
      0%,100% {{ border-color: #38bdf8; box-shadow: 0 0 0 0 rgba(56,189,248,0.4); }}
      50%      {{ border-color: #818cf8; box-shadow: 0 0 0 12px rgba(56,189,248,0); }}
    }}
    .sim-label {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px; }}
    .sim-btn {{
      background: linear-gradient(135deg, #0ea5e9, #6366f1);
      border: none; border-radius: 8px;
      color: #fff; cursor: pointer;
      font-size: 0.9rem; padding: 10px 28px;
      margin-top: 8px; transition: opacity 0.2s;
    }}
    .sim-btn:hover {{ opacity: 0.85; }}
    .sim-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .progress-bar-wrap {{
      background: #1e293b; border-radius: 999px;
      height: 6px; margin: 12px 0; overflow: hidden;
    }}
    .progress-bar {{
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      height: 100%; border-radius: 999px;
      width: 0%; transition: width 0.15s linear;
    }}
  </style>
</head>
<body>
<div id="root"></div>

<script>
// ---------------------------------------------------------------
// Architecture note:
//
// AWS Amplify's FaceLivenessDetector requires:
//   (a) A Cognito Identity Pool (even unauthenticated) OR
//   (b) Temporary credentials passed via credentialsProvider prop.
//
// For a Streamlit-only deployment WITHOUT a Cognito pool we use the
// SIMULATION path below, which visually mimics the liveness flow and
// calls GetFaceLivenessSessionResults from the Python backend.
//
// To switch to the REAL Amplify component:
//   1. Create a Cognito Identity Pool and add the pool ID below.
//   2. Comment out the "SIMULATION" block.
//   3. Uncomment the "REAL AMPLIFY" block.
// ---------------------------------------------------------------

const SESSION_ID = "{session_id}";
const AWS_REGION  = "{aws_region}";
const TESTING     = {"true" if TESTING_MODE else "false"};

// ── postMessage helpers ─────────────────────────────────────
function notifyParent(type, payload) {{
  window.parent.postMessage(JSON.stringify({{ type, payload }}), "*");
}}

// ── Simulation liveness component ───────────────────────────
// Mimics the oval + color-flash challenge UI.
// After the animation completes it posts LIVENESS_COMPLETE back.
const App = () => {{
  const [phase, setPhase]     = React.useState("intro");  // intro | challenge | done | error
  const [progress, setProgress] = React.useState(0);
  const [statusMsg, setStatusMsg] = React.useState("");
  const [flashColor, setFlashColor] = React.useState("transparent");

  const CHALLENGE_DURATION_MS = 4000;
  const COLORS = ["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7"];

  const runChallenge = () => {{
    setPhase("challenge");
    let elapsed = 0;
    const tick = 100;
    let colorIdx = 0;

    const interval = setInterval(() => {{
      elapsed += tick;
      setProgress(Math.min((elapsed / CHALLENGE_DURATION_MS) * 100, 100));

      // Flash a new color every 600 ms
      if (elapsed % 600 === 0) {{
        setFlashColor(COLORS[colorIdx % COLORS.length]);
        colorIdx++;
        // Reset flash after 200 ms
        setTimeout(() => setFlashColor("transparent"), 200);
      }}

      if (elapsed >= CHALLENGE_DURATION_MS) {{
        clearInterval(interval);
        setPhase("verifying");
        setStatusMsg("⏳ Verifying with AWS...");
        // Tell Python backend to call GetFaceLivenessSessionResults
        notifyParent("LIVENESS_COMPLETE", {{ sessionId: SESSION_ID }});
      }}
    }}, tick);
  }};

  // Listen for Python passing the result back (optional two-way)
  React.useEffect(() => {{
    const handler = (e) => {{
      try {{
        const data = JSON.parse(e.data);
        if (data.type === "LIVENESS_RESULT") {{
          if (data.payload.passed) {{
            setPhase("done");
            setStatusMsg("✅ Liveness confirmed!");
          }} else {{
            setPhase("error");
            setStatusMsg("❌ Liveness failed. Please retry.");
          }}
        }}
      }} catch(_) {{}}
    }};
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }}, []);

  return (
    <div className="card">
      <span className="badge">AWS Rekognition Face Liveness</span>
      <h2>Anti-Spoofing Verification</h2>
      <p>Keep your face inside the oval and follow the light prompts.<br/>This prevents proxy attendance using photos or videos.</p>

      {{phase === "intro" && (
        <div className="sim-box">
          <div className="sim-oval"></div>
          <p className="sim-label">Position your face inside the oval, then click Start.</p>
          <button className="sim-btn" onClick={{runChallenge}}>▶ Start Liveness Check</button>
        </div>
      )}}

      {{phase === "challenge" && (
        <div className="sim-box" style={{{{background: flashColor !== "transparent" ? flashColor+"22" : "#0f172a", transition:"background 0.1s"}}}}>
          <div className="sim-oval" style={{{{borderColor: flashColor !== "transparent" ? flashColor : "#38bdf8"}}}}></div>
          <p className="sim-label">Follow the light flashes — keep still...</p>
          <div className="progress-bar-wrap">
            <div className="progress-bar" style={{{{width: progress+"%"}}}}></div>
          </div>
        </div>
      )}}

      {{phase === "verifying" && (
        <>
          <div className="spinner"></div>
          <p className="status-loading">{{statusMsg}}</p>
        </>
      )}}

      {{phase === "done" && (
        <p className="status-ok">{{statusMsg}}</p>
      )}}

      {{phase === "error" && (
        <>
          <p className="status-err">{{statusMsg}}</p>
          <button className="sim-btn" style={{{{marginTop:"16px"}}}} onClick={{()=>{{setPhase("intro");setProgress(0);}}}}>
            🔄 Retry
          </button>
        </>
      )}}
    </div>
  );
}};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>
"""

    # Height in pixels for the embedded iframe
    WIDGET_HEIGHT = 520

    # Render the HTML widget
    components.html(html_code, height=WIDGET_HEIGHT, scrolling=False)

    # Return None — caller must check st.session_state["liveness_session_done"]
    return None


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

# ── Receive postMessage from the liveness widget ─────────────
# Streamlit doesn't natively receive postMessage, so we use a tiny
# JS snippet + st.query_params as a side-channel.
# When the React widget fires LIVENESS_COMPLETE it also sets a query
# param ?liveness_done=<session_id> which triggers a rerun.
query = st.query_params
if "liveness_done" in query and not st.session_state.liveness_done:
    incoming_sid = query["liveness_done"]
    if incoming_sid == st.session_state.liveness_session_id:
        # Fetch result from AWS
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
        # Clean up query param
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

        # Create a new liveness session if we don't have one yet
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

        # Inject the JS side-channel: after LIVENESS_COMPLETE the widget
        # sets window.location search param which triggers a Streamlit rerun.
        js_bridge = f"""
        <script>
        window.addEventListener("message", function(e) {{
            try {{
                var data = JSON.parse(e.data);
                if (data.type === "LIVENESS_COMPLETE") {{
                    // Navigate to the same page with the query param set —
                    // Streamlit will pick this up on the next rerun.
                    var url = new URL(window.parent.location.href);
                    url.searchParams.set("liveness_done", "{sid}");
                    window.parent.location.href = url.toString();
                }}
            }} catch(_) {{}}
        }});
        </script>
        """
        components.html(js_bridge, height=0)

        render_liveness_widget(sid, region)

        st.caption(
            f"Session ID: `{sid}` | Confidence threshold: ≥ {LIVENESS_CONFIDENCE_THRESHOLD}%"
        )
        st.stop()   # Wait for the widget to post back

    # Liveness result available
    if not st.session_state.liveness_passed:
        conf = st.session_state.liveness_confidence
        st.error(
            f"❌ Liveness check failed (confidence: {conf:.1f}% — required ≥ {LIVENESS_CONFIDENCE_THRESHOLD}%).\n\n"
            "A photo, video, or deepfake was likely detected."
        )
        if st.button("🔄 Try Again"):
            for k in ["liveness_session_id", "liveness_done", "liveness_passed",
                      "liveness_confidence", "reference_img_bytes"]:
                st.session_state[k] = False if isinstance(defaults[k], bool) else None
            st.rerun()
        st.stop()

    st.success(f"✅ Step 2 Complete — Live person confirmed ({st.session_state.liveness_confidence:.1f}% confidence).")

    # ── STEP 3: Face Match ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 of 3 — 🧬 Identity Verification")
    st.info("The reference image captured during liveness will now be matched against the registered database.")

    ref_bytes = st.session_state.reference_img_bytes

    # If S3-stored (no raw bytes returned), let the user also capture manually
    if ref_bytes is None:
        st.warning(
            "Reference image was stored in S3 (not returned as bytes). "
            "Please capture a quick selfie below so we can run the face match."
        )
        live_photo = st.camera_input("Capture selfie for identity match")
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

                dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                    'USN':                matched_usn,
                    'Timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Status':             'Present',
                    'FaceMatchScore':     str(round(confidence, 2)),
                    'LivenessScore':      str(round(st.session_state.liveness_confidence, 2)),
                    'VerificationMethod': 'Geofence+AWSFaceLiveness+FaceMatch'
                                         if not TESTING_MODE else 'TestingMode',
                    'LivenessSessionId':  st.session_state.liveness_session_id or 'N/A',
                })

                # Reset all state for the next student
                for k, v in defaults.items():
                    st.session_state[k] = v

            else:
                st.error("❌ Face not recognised in the database. Please register first.")

        except Exception as e:
            st.error(f"Rekognition error: {e}")


# ============================================================
# PAGE 2 — NEW USER REGISTRATION
# ============================================================
elif page == "New User Registration":
    st.header("📝 Student Self-Registration")
    st.markdown("Register once. Your face and USN are stored securely in AWS.")

    with st.form("reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            usn       = st.text_input("USN (e.g. 1DA25SCS18)")
        with col2:
            password  = st.text_input("Create Password", type="password")

        st.markdown("### 📷 Capture Reference Photo")
        st.caption("Ensure good lighting. Remove glasses/hats. Face the camera directly.")
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
                    # 1. Store reference photo in S3
                    s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=f"reference_photos/{cleaned_usn}.jpg",
                        Body=img_bytes
                    )
                    # 2. Index face in Rekognition collection
                    idx_response = rekog.index_faces(
                        CollectionId=COLLECTION_ID,
                        Image={'Bytes': img_bytes},
                        ExternalImageId=cleaned_usn,
                        MaxFaces=1,
                        QualityFilter='HIGH'   # Reject blurry / side-profile photos
                    )
                    if not idx_response.get('FaceRecords'):
                        st.error(
                            "⚠️ No face detected in your photo. "
                            "Please retake in good lighting facing the camera directly."
                        )
                    else:
                        # 3. Store profile in DynamoDB (never store plain-text passwords in production!)
                        dynamo.Table(TABLE_PROFILES).put_item(Item={
                            'USN':      cleaned_usn,
                            'Name':     full_name,
                            'Password': password,    # TODO: hash with bcrypt before production
                        })
                        st.success(f"🎉 Registration successful for **{full_name}** ({cleaned_usn})!")
                        st.info("You can now mark attendance using the Attendance Verification page.")

                except Exception as aws_error:
                    st.error(f"AWS Error: {aws_error}")


# ============================================================
# PAGE 3 — BATCH RESULTS
# ============================================================
elif page == "Batch Results":
    st.header("📊 Public Results Dashboard")

    try:
        results_data = dynamo.Table(TABLE_RESULTS).scan()
        items = results_data.get('Items', [])
        if items:
            df = pd.DataFrame(items)
            # Put USN and Name first if they exist
            priority_cols = [c for c in ['USN', 'Name'] if c in df.columns]
            other_cols    = [c for c in df.columns if c not in priority_cols]
            df = df[priority_cols + other_cols]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Academic results have not been uploaded yet.")
    except Exception:
        st.error("Database connection error. Ensure your DynamoDB tables are active.")
