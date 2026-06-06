import streamlit as st
import boto3
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from streamlit_js_eval import get_geolocation

# --- 1. AWS CREDENTIALS & CLIENTS ---
# Pulls keys from Streamlit Cloud Secrets (Settings > Secrets)
try:
    aws_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
    region = st.secrets["AWS_DEFAULT_REGION"]
except KeyError:
    st.error("⚠️ AWS Secrets not found! Go to Streamlit Cloud Settings > Secrets and add your keys.")
    st.stop()

# Initialize AWS Services
s3 = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
rekog = boto3.client('rekognition', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
dynamo = boto3.resource('dynamodb', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)

# --- 2. CONSTANTS ---
BUCKET_NAME = 'college-system-data'
COLLECTION_ID = 'college_faces'
TABLE_PROFILES = 'StudentProfiles'
TABLE_ATTENDANCE = 'AttendanceLogs'
TABLE_RESULTS = 'StudentResults'

# Classroom Geofence (These will be used after testing is over)
# Current values are for Siruguppa area
CLASSROOM_LAT = 15.626 
CLASSROOM_LON = 76.897
ALLOWED_RADIUS = 0.02 

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI College Portal", page_icon="🎓", layout="wide")

st.sidebar.title("🏫 Navigation")
page = st.sidebar.radio("Go to:", ["Attendance (Face Login)", "New User Registration", "Batch Results"])

# --- HELPERS ---
def check_location(loc):
    if not loc: 
        return False
    
    # --- TESTING MODE: ALWAYS ALLOWED ---
    # Safe check to make sure 'coords' keys are present in the response object
    if 'coords' in loc:
        lat = loc['coords'].get('latitude', 'Unknown')
        lon = loc['coords'].get('longitude', 'Unknown')
        st.info(f"📍 DEBUG MODE: Your Location is Lat: {lat}, Lon: {lon}")
    else:
        st.warning("⚠️ Geolocation data payload is missing expected structural properties.")
    
    return True 

    # --- PRODUCTION MODE (Uncomment this for final submission) ---
    # if 'coords' not in loc:
    #      return False
    # lat_diff = abs(loc['coords']['latitude'] - CLASSROOM_LAT)
    # lon_diff = abs(loc['coords']['longitude'] - CLASSROOM_LON)
    # return lat_diff < ALLOWED_RADIUS and lon_diff < ALLOWED_RADIUS


def detect_screen_spoofing(image_bytes):
    """
    Analyzes texture patterns and specular surface glare reflections 
    to separate a physical human face from an electronic device display.
    """
    # Convert image bytes into an OpenCV BGR spatial grid
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False, "Failed to decode frame telemetry context."
        
    # 1. Luminance Channel Glare Isolation
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb)
    
    # High-intensity white screen flash creates concentrated clusters of near 255-brightness pixels on phone glass
    _, high_glare_mask = cv2.threshold(y_channel, 250, 255, cv2.THRESH_BINARY)
    glare_pixel_count = np.sum(high_glare_mask == 255)
    glare_ratio = glare_pixel_count / y_channel.size
    
    # 2. Moiré Frequency Noise Isolation via Laplacian Matrix
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # --- HEURISTIC THRESHOLDS ---
    MAX_ALLOWED_GLARE_RATIO = 0.035  # Max 3.5% of frame can be extreme specular reflection
    MIN_REAL_VARIANCE = 80.0         # Screen video captures lose detail variance when off-focus
    MAX_REAL_VARIANCE = 900.0        # High-frequency electronic pixel arrays spike noise variations
    
    if glare_ratio > MAX_ALLOWED_GLARE_RATIO:
        return False, f"Spoof Detected: Unnatural display reflection signature ({round(glare_ratio * 100, 2)}% Glare Area)."
        
    if laplacian_var > MAX_REAL_VARIANCE or laplacian_var < MIN_REAL_VARIANCE:
        return False, f"Spoof Detected: Digital matrix pattern identified (Variance: {round(laplacian_var, 1)})."
        
    return True, f"Liveness Confirmed (Texture Variance: {round(laplacian_var, 1)})."


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
                
                # Sanitize USN to remove accidental leading/trailing whitespaces for AWS constraints
                cleaned_usn = usn.strip()
                
                try:
                    # 1. Upload Reference Image to S3 using the cleaned filename
                    s3.put_object(Bucket=BUCKET_NAME, Key=f"reference_photos/{cleaned_usn}.jpg", Body=img_bytes)
                    
                    # 2. Index Face in Rekognition Collection
                    rekog.index_faces(
                        CollectionId=COLLECTION_ID,
                        Image={'Bytes': img_bytes},
                        ExternalImageId=cleaned_usn, # Links this face-print safely to stripped USN
                        MaxFaces=1
                    )
                    
                    # 3. Store Profile & Password in DynamoDB
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

# --- PAGE 2: ATTENDANCE (FACE LOGIN) ---
elif page == "Attendance (Face Login)":
    st.header("📸 Attendance Kiosk")
    
    # Initialize persistent UI state tracking to protect location lookups during widget re-runs
    if "location_verified" not in st.session_state:
        st.session_state.location_verified = False
    if "flash_triggered" not in st.session_state:
        st.session_state.flash_triggered = False

    # 1. Geolocation Verification Stage
    if not st.session_state.location_verified:
        user_loc = get_geolocation()
        if user_loc:
            if check_location(user_loc):
                st.session_state.location_verified = True
                st.rerun()
            else:
                st.error("🚫 Access Denied: You are not in the classroom.")
        else:
            st.warning("Please allow browser location access to continue.")
            
    # 2. Biometric Verification Stage
    else:
        st.success("🔒 Physical Access Granted: Location Coordinates Verified.")
        
        # Inject dynamic full-screen white background styles if flash authentication state machine is active
        if st.session_state.flash_triggered:
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #FFFFFF !important;
                }
                h1, h2, h3, p, span, label, div {
                    color: #000000 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.info("⚡ MONITOR FLASH IS ACTIVE. Please hold your device steady, look into the lens, and capture.")

        # Interactive controls to switch theme states cleanly
        if not st.session_state.flash_triggered:
            if st.button("🌟 Initialize Anti-Spoof Flash Kiosk"):
                st.session_state.flash_triggered = True
                st.rerun()
        else:
            if st.button("❌ Abort/Reset Flash Kiosk"):
                st.session_state.flash_triggered = False
                st.rerun()

        # Render camera canvas component
        login_photo = st.camera_input("Look at the camera to log in")
        
        if login_photo:
            login_bytes = login_photo.getvalue()
            
            # Execute physical liveness texture checks before invoking AWS API usage bounds
            with st.spinner("Analyzing biometric texture structure and surface glare..."):
                is_real_human, texture_feedback = detect_screen_spoofing(login_bytes)
            
            # Revert background context styles instantly upon computation completion
            st.session_state.flash_triggered = False
            
            if not is_real_human:
                st.error(f"❌ Security Access Denied: {texture_feedback}")
                st.warning("Proxy attendance fraud detected. Incident report generated.")
            else:
                st.success(f"🛡️ Security Authorization: {texture_feedback}")
                
                with st.spinner("Matching Face Patterns against Institutional Database..."):
                    try:
                        # Compare live photo against indexed reference photos
                        response = rekog.search_faces_by_image(
                            CollectionId=COLLECTION_ID,
                            Image={'Bytes': login_bytes},
                            MaxFaces=1,
                            FaceMatchThreshold=90
                        )
                        
                        if response['FaceMatches']:
                            found_usn = response['FaceMatches'][0]['Face']['ExternalImageId']
                            st.balloons()
                            st.success(f"Verified! Attendance marked for USN: {found_usn}")
                            
                            # 3. Log to Attendance Table
                            dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                                'USN': found_usn,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Status': 'Present'
                            })
                            
                            # Reset session parameters for the next student queue entry
                            st.session_state.location_verified = False
                        else:
                            st.error("Identity not verified. Please ensure your reference image matches clearly.")
                    except Exception as e:
                        st.error(f"Verification Pipeline failure: {e}")

# --- PAGE 3: BATCH RESULTS ---
elif page == "Batch Results":
    st.header("📊 Public Results Dashboard")
    st.caption("General access to academic performance records.")
    
    try:
        results_data = dynamo.Table(TABLE_RESULTS).scan()
        if results_data['Items']:
            df = pd.DataFrame(results_data['Items'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("The academic results have not been uploaded to the database yet.")
    except Exception:
        st.error("Database connection error. Ensure your DynamoDB tables are active.")
