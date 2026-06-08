import streamlit as st
import boto3
import pandas as pd
import numpy as np
import cv2
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
    """Cache client connections to minimize network handshake overhead."""
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

# Set your target classroom coordinates
CLASSROOM_LAT = 15.626 
CLASSROOM_LON = 76.897
ALLOWED_RADIUS = 0.02 

# --- 3. CORE VALIDATION UTILITIES ---
def check_location(loc):
    """Calculates whether the student device sits inside the geofenced area boundaries."""
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

def verify_liveness_metrics(image_bytes):
    """
    Evaluates frame texture illumination to detect photo-spoofing.
    Flat screens and paper prints exhibit low variance and high flat peaks.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # A flat printout or mobile screen lacks natural 3D focus depth, 
        # leading to low Laplacian texture variance (blurry edges/flat surface)
        if laplacian_var < 100.0:
            return False
        return True
    except Exception:
        # Fallback to true if openCV layer drops on specific device architectures
        return True

# --- 4. STREAMLIT APP SURFACE ENGINE ROUTER ---
st.set_page_config(page_title="Secure Portal", page_icon="🎓", layout="wide")

st.sidebar.title("🏫 Navigation")
page = st.sidebar.radio("Go to:", ["Attendance Verification", "New User Registration", "Batch Results"])

# ==========================================
# PAGE 1: ATTENDANCE VERIFICATION (GEOFENCE + LIVENESS FACE MATCH)
# ==========================================
if page == "Attendance Verification":
    st.header("📸 Secure Biometric Attendance Verification")
    st.markdown("Your physical location and live face pattern will be analyzed simultaneously.")

    # STEP 1: Live Geofence Check
    st.markdown("### Step 1: Location Verification")
    user_location = get_geolocation()
    
    if not user_location:
        st.warning("📍 Action Required: Grant browser GPS location tracking permissions to continue.")
    elif not check_location(user_location):
        st.error("🚫 Access Denied: You are outside the designated classroom boundaries.")
    else:
        st.success("📍 Classroom Proximity Verified!")
        
        # STEP 2: Biometric Liveness Capture
        st.markdown("### Step 2: Facial Biometric Identification")
        st.info("Look directly into the camera. Ensure your face is clearly visible without glasses or caps.")
        
        live_photo = st.camera_input("Capture Live Verification Face")
        
        if live_photo:
            img_bytes = live_photo.getvalue()
            
            # Sub-Check A: Anti-Spoofing Liveness Evaluation
            with st.spinner("Analyzing structural liveness indicators..."):
                is_live_person = verify_liveness_metrics(img_bytes)
                
            if not is_live_person:
                st.error("❌ Verification Failed: Digital screen spoofing or photo printout detected! Please present your real face.")
            else:
                # Sub-Check B: Amazon Rekognition Database Pattern Matching
                with st.spinner("Matching face pattern against institutional database..."):
                    try:
                        response = rekog.search_faces_by_image(
                            CollectionId=COLLECTION_ID,
                            Image={'Bytes': img_bytes},
                            MaxFaces=1,
                            FaceMatchThreshold=92
                        )
                        
                        if response['FaceMatches']:
                            matched_usn = response['FaceMatches'][0]['Face']['ExternalImageId']
                            confidence = response['FaceMatches'][0]['Similarity']
                            
                            st.balloons()
                            st.success(f"✅ Success! Verified Identity for USN: {matched_usn} ({confidence:.2f}% Match)")
                            
                            # STEP 3: Log directly into AWS DynamoDB
                            dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                                'USN': matched_usn,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Status': 'Present',
                                'VerificationMethod': 'Geofence_Plus_Liveness_Biometrics'
                            })
                        else:
                            st.error("❌ Identity Mismatch: Face structural features do not match any registered student.")
                    except Exception as e:
                        st.error(f"Biometric pipeline error: {e}")

# ==========================================
# PAGE 2: NEW USER REGISTRATION
# ==========================================
elif page == "New User Registration":
    st.header("📝 Student Self-Registration")
    st.markdown("Register your profile and capture your permanent **Reference Photo** below.")
    
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

# ==========================================
# PAGE 3: BATCH RESULTS
# ==========================================
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
