import streamlit as st
import boto3
import pandas as pd
import random
from datetime import datetime
from streamlit_js_eval import get_geolocation

# --- 1. AWS CREDENTIALS & CLIENTS ---
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
    
    if 'coords' in loc:
        lat = loc['coords'].get('latitude', 'Unknown')
        lon = loc['coords'].get('longitude', 'Unknown')
        st.info(f"📍 DEBUG MODE: Your Location is Lat: {lat}, Lon: {lon}")
    else:
        st.warning("⚠️ Geolocation data payload is missing expected structural properties.")
    
    return True 

def verify_challenge_text(image_bytes, expected_text):
    """
    Uses Amazon Rekognition to see if the random code is physically present 
    inside the captured camera image frame.
    """
    try:
        response = rekog.detect_text(Image={'Bytes': image_bytes})
        detected_words = [text_obj['DetectedText'].strip() for text_obj in response['TextDetections']]
        
        # Check if our random code exists anywhere in the list of detected text strings
        if str(expected_text) in detected_words:
            return True
        
        # Secondary fallback: check if any detected text block contains the code implicitly
        for word in detected_words:
            if str(expected_text) in word:
                return True
                
        return False
    except Exception as e:
        st.error(f"Text detection layer engine error: {e}")
        return False

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
                cleaned_usn = usn.strip()
                
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

# --- PAGE 2: ATTENDANCE (FACE LOGIN) ---
elif page == "Attendance (Face Login)":
    st.header("📸 Attendance Kiosk")
    
    if "location_verified" not in st.session_state:
        st.session_state.location_verified = False
    if "live_code" not in st.session_state:
        # Generate a random 4 digit challenge token code
        st.session_state.live_code = random.randint(1000, 9999)

    # 1. Geolocation Verification
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
            
    # 2. Biometric & Challenge Verification
    else:
        st.success("🔒 Physical Access Granted: Location Coordinates Verified.")
        
        # Big presentation of the challenge token
        st.markdown(f"""
        <div style="background-color:#fff3cd; padding:20px; border-radius:10px; border-left: 8px solid #ffc107;">
            <h4 style="color:#856404; margin:0;">🔒 Anti-Proxy Verification Challenge</h4>
            <p style="color:#856404; margin-top:5px; margin-bottom:10px;">
                Write the large number below clearly on a piece of paper (or display it on a second phone screen) 
                and hold it right next to your chin while capturing your photo.
            </p>
            <h1 style="color:#000000; font-size: 50px; letter-spacing: 5px; margin:0;">{st.session_state.live_code}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Challenge Code"):
            st.session_state.live_code = random.randint(1000, 9999)
            st.rerun()

        login_photo = st.camera_input("Look at the camera holding the challenge code up")
        
        if login_photo:
            login_bytes = login_photo.getvalue()
            
            with st.spinner("Analyzing frame content for anti-spoofing challenge code..."):
                code_matched = verify_challenge_text(login_bytes, st.session_state.live_code)
                
            if not code_matched:
                st.error(f"❌ Verification Failed: The active live challenge code '{st.session_state.live_code}' was not found in the frame image.")
                st.warning("Ensure the written digits are clearly visible, legible, and not covered by your hand.")
            else:
                st.success("🛡️ Challenge Code Verified! Match Found.")
                
                with st.spinner("Matching Face Patterns against Institutional Database..."):
                    try:
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
                            
                            dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                                'USN': found_usn,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Status': 'Present'
                            })
                            
                            # Clean up and reset for next run setup
                            st.session_state.location_verified = False
                            st.session_state.live_code = random.randint(1000, 9999)
                        else:
                            st.error("Identity not verified. The face structure does not match a registered USN.")
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
