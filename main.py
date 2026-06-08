import streamlit as st
import boto3
import pandas as pd
import random
import hmac
import hashlib
import time
import qrcode
from io import BytesIO
from datetime import datetime
from streamlit_js_eval import get_geolocation

# --- 1. AWS CREDENTIALS & INITIALIZATION ---
try:
    aws_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
    region = st.secrets["AWS_DEFAULT_REGION"]
except KeyError:
    st.error("⚠️ AWS Secrets not found! Go to Streamlit Cloud Settings > Secrets and add your keys.")
    st.stop()

@st.cache_resource
def init_aws_resources():
    """Cache client initializations to save network roundtrips on periodic script reruns."""
    s3_client = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    rekog_client = boto3.client('rekognition', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    dynamo_res = boto3.resource('dynamodb', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
    return s3_client, rekog_client, dynamo_res

s3, rekog, dynamo = init_aws_resources()

# --- 2. CONSTANTS & GEOLOCATION PARAMETERS ---
BUCKET_NAME = 'college-system-data'
COLLECTION_ID = 'college_faces'
TABLE_PROFILES = 'StudentProfiles'
TABLE_ATTENDANCE = 'AttendanceLogs'
TABLE_RESULTS = 'StudentResults'

CLASSROOM_LAT = 15.626 
CLASSROOM_LON = 76.897
ALLOWED_RADIUS = 0.02 

# Security configurations for token validation
HMAC_SECRET_KEY = "SECRET_COLLEGE_PORTAL_SIGNING_SALT_KEY"
QR_EXPIRY_SECONDS = 40  # The QR code signature automatically rotates every 15 seconds

# --- 3. CRYPTOGRAPHIC HANDSHAKE UTILITIES ---
def generate_secure_token():
    """Generates a unique dynamic token valid for the current time block."""
    time_block = int(time.time() // QR_EXPIRY_SECONDS)
    message = f"classroom_kiosk_block_{time_block}".encode('utf-8')
    token = hmac.new(HMAC_SECRET_KEY.encode('utf-8'), message, hashlib.sha256).hexdigest()
    return token, time_block

def verify_secure_token(token_to_test, time_block_used):
    """Validates the client-submitted token, accepting a 1-block window lag tolerance."""
    current_block = int(time.time() // QR_EXPIRY_SECONDS)
    for block in [current_block, current_block - 1]:
        message = f"classroom_kiosk_block_{block}".encode('utf-8')
        expected = hmac.new(HMAC_SECRET_KEY.encode('utf-8'), message, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, token_to_test):
            return True
    return False

def check_location(loc):
    """Calculates whether the reporting student device resides inside the geofenced area boundaries."""
    if not loc or 'coords' not in loc:
        return False
    lat = loc['coords'].get('latitude')
    lon = loc['coords'].get('longitude')
    if lat is None or lon is None:
        return False
        
    st.info(f"📍 Location Coordinates Captured - Lat: {lat:.4f}, Lon: {lon:.4f}")
    
    lat_min, lat_max = CLASSROOM_LAT - ALLOWED_RADIUS, CLASSROOM_LAT + ALLOWED_RADIUS
    lon_min, lon_max = CLASSROOM_LON - ALLOWED_RADIUS, CLASSROOM_LON + ALLOWED_RADIUS
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


# --- 4. STREAMLIT APP SURFACE ENGINE ROUTER ---
# Read query arguments to detect whether the user is a desktop kiosk display screen or a smartphone browser
query_params = st.query_params

# ==========================================
# BRANCH ROUTE A: STUDENT MOBILE MODE (?mode=student)
# ==========================================
if query_params.get("mode") == "student":
    st.set_page_config(page_title="Mobile Check-In", page_icon="📱", layout="centered")
    st.header("📱 Secure Student Mobile Check-In")
    
    scanned_token = query_params.get("token")
    scanned_block = query_params.get("block")
    
    if not scanned_token or not scanned_block:
        st.error("🚫 Invalid Verification Link. Please scan the active classroom board image token directly.")
        st.stop()
        
    # Phase 1: Verify QR Code Token Validity
    if not verify_secure_token(scanned_token, scanned_block):
        st.error("⏰ Session Timed Out! The scanned QR code has expired. Please look up and scan the fresh code on the screen.")
        st.stop()
        
    st.success("🔒 Security Token Cryptographically Validated.")

    # Phase 2: Verify Real-Time GPS Proximity Range
    st.markdown("### Step 1: Verification of Classroom Vicinity Presence")
    user_location = get_geolocation()
    
    if not user_location:
        st.warning("📍 Action Required: Grant browser GPS location tracking access settings clearance to execute your session.")
    elif not check_location(user_location):
        st.error("🚫 Proxy Prevention Error: You are outside the classroom boundary radius limit range parameters.")
    else:
        st.success("📍 Classroom Proximity Verified!")
        
        # Phase 3: Identity Logging Authentication
        st.markdown("### Step 2: Account Login Confirmation")
        with st.form("mobile_signin_form"):
            student_usn = st.text_input("Enter your USN (e.g., 1DA25SCS18)").strip().upper()
            student_pass = st.text_input("Enter Password Profile Keyphrase", type="password")
            submit_checkin = st.form_submit_button("Verify & Sign Attendance Log")
            
        if submit_checkin:
            if student_usn and student_pass:
                with st.spinner("Writing check-in validation log record..."):
                    try:
                        profile_info = dynamo.Table(TABLE_PROFILES).get_item(Key={'USN': student_usn})
                        
                        if 'Item' in profile_info and profile_info['Item']['Password'] == student_pass:
                            dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                                'USN': student_usn,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Status': 'Present',
                                'VerificationMethod': 'DualDevice_QR_Sync_GPS'
                            })
                            st.balloons()
                            st.success(f"✅ Roll Call Successful! Attendance documented for USN: {student_usn}. You may exit your mobile web browser safely.")
                        else:
                            st.error("❌ Sign-in Refused: The provided USN or password record mapping is mismatched.")
                    except Exception as transaction_err:
                        st.error(f"Database logging operation fault occurred: {transaction_err}")
            else:
                st.warning("⚠️ Complete both input criteria entries before submitting authentication requests.")

# ==========================================
# BRANCH ROUTE B: CORE DESKTOP KIOSK PORTAL APPLICATION SURFACE
# ==========================================
else:
    st.set_page_config(page_title="AI College Portal", page_icon="🎓", layout="wide")

    st.sidebar.title("🏫 Navigation")
    page = st.sidebar.radio("Go to:", ["Attendance (Face Login)", "New User Registration", "Batch Results"])

    # --- PAGE 1: NEW USER REGISTRATION ---
    if page == "New User Registration":
        st.header("📝 Student Self-Registration")
        st.markdown("Register your profile and take your permanent **Reference Photo** below.")
        
        with st.form("reg_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name")
                usn = st.text_input("USN (e.g., 1DA25SCS18)")
            with col2:
                password = st.text_input("Create Password", type="password")
                
            st.write("### Step 2: Capture Reference Photo")
            reg_photo = st.camera_input("Take Official Profile Photo")
            submit = st.form_submit_button("Register Student")
            
        if submit:
            if full_name and usn and password and reg_photo:
                with st.spinner("Uploading Profile to AWS..."):
                    img_bytes = reg_photo.getvalue()
                    cleaned_usn = usn.strip().upper()
                    
                    try:
                        s3.put_object(Bucket=BUCKET_NAME, Key=f"reference_photos/{cleaned_usn}.jpg", Body=img_bytes)
                        rekog.index_faces(
                            CollectionId=COLLECTION_ID,
                            Image={'Bytes': img_bytes},
                            ExternalImageId=cleaned_usn,
                            MaxFaces=1
                        )
                        dynamo.Table(TABLE_PROFILES).put_item(Item={
                            'USN': cleaned_usn,
                            'Name': full_name,
                            'Password': password
                        })
                        st.success(f"Registration successful for {full_name} ({cleaned_usn})!")
                    except Exception as aws_error:
                        st.error(f"AWS Error: {aws_error}")
            else:
                st.error("Please fill all fields and capture your photo.")

    # --- PAGE 2: ATTENDANCE (KIOSK VIEW) ---
    elif page == "Attendance (Face Login)":
        st.header("📸 Anti-Proxy Attendance Kiosk Terminal")
        
        layout_col_left, layout_col_right = st.columns([2, 1])
        
        with layout_col_left:
            st.write("### 📌 Instructions for Live Class Roll Call Verification")
            st.markdown("""
            1. Pull out your **personal mobile device** inside the lecture room boundaries.
            2. Open your smartphone camera app or an official web scanner engine.
            3. Point your camera at the **dynamic rotating code matrix card** visible on the right block panel.
            4. Grant your mobile browser instant clearance permission requests to evaluate your real-world spatial location coordinates.
            5. Provide your authentic profile USN and password security credentials to close your confirmation handshake.
            
            *⏰ Note: This security card block automatically shifts values every **15 seconds**. Remote users attempting to bypass check-ins using shared video loops or chat image captures will be denied authorization.*
            """)
            
            if st.button("🔄 Manually Cycle Code Token Matrix"):
                st.rerun()

        with layout_col_right:
            # Generate temporary rotating authorization hash string parameters
            current_token, time_block_id = generate_secure_token()
            
            # 🔧 CRITICAL PRODUCTION TASK FOR CLOUD DEPLOYMENT:
            # If deploying live on Streamlit Cloud, update this variable to point to your live URL domain string.
            # Example: base_app_url = "https://your-college-portal.streamlit.app"
            # If running locally on local Wi-Fi, use your laptop's Local Network IP instead of localhost:
            # Example: base_app_url = "http://192.168.1.45:8501"
            base_app_url = "https://ty7896.streamlit.app" 
            
            target_scan_url = f"{base_app_url}/?mode=student&token={current_token}&block={time_block_id}"
            
            # Render a high-density black/white image array wrapper card
            qr_engine = qrcode.QRCode(version=1, box_size=10, border=2)
            qr_engine.add_data(target_scan_url)
            qr_engine.make(fit=True)
            
            qr_image = qr_engine.make_image(fill_color="black", back_color="white")
            image_buffer = BytesIO()
            qr_image.save(image_buffer, format="PNG")
            byte_payload = image_buffer.getvalue()
            
            st.image(byte_payload, caption=f"Active Dynamic Handshake Signature Token.", use_container_width=False, width=320)
            
        # UI REFRESH HEARTBEAT TICKER MODULE
        # Executes non-blocking client-side webpage browser loops to force the Kiosk view to redraw automatically
        st.components.v1.html(
            f"""
            <script>
                setTimeout(function(){{
                    window.parent.location.reload();
                }}, {QR_EXPIRY_SECONDS * 1000});
            </script>
            """,
            height=0
        )

    # --- PAGE 3: BATCH RESULTS ---
    elif page == "Batch Results":
        st.header("📊 Public Results Dashboard")
        st.caption("General access to academic performance records.")
        
        try:
            results_data = dynamo.Table(TABLE_RESULTS).scan()
            if results_data.get('Items'):
                df = pd.DataFrame(results_data['Items'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("The academic results have not been uploaded to the database yet.")
        except Exception:
            st.error("Database connection error. Ensure your DynamoDB tables are active.")
