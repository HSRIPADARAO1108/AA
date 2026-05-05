import streamlit as st
import boto3
import pandas as pd
from datetime import datetime
from streamlit_js_eval import get_geolocation

# --- 1. CLOUD CONFIGURATION & SECURITY ---
# This section pulls your keys securely from the Streamlit Cloud "Secrets" dashboard.
# Do NOT hardcode your keys here. Keep this code exactly as is.
try:
    aws_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
    region = st.secrets["AWS_DEFAULT_REGION"]
except KeyError:
    st.error("⚠️ AWS Secrets not found! Go to Streamlit Cloud Settings > Secrets and add your keys.")
    st.stop()

# Initialize AWS Clients
s3 = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
rekog = boto3.client('rekognition', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)
dynamo = boto3.resource('dynamodb', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, region_name=region)

# --- 2. PROJECT CONSTANTS ---
BUCKET_NAME = 'college-system-data' # Ensure this matches your S3 bucket name
COLLECTION_ID = 'college_faces'     # Ensure you created this in AWS CloudShell
TABLE_PROFILES = 'StudentProfiles'
TABLE_ATTENDANCE = 'AttendanceLogs'
TABLE_RESULTS = 'StudentResults'

# GEOFENCING: Set these to your classroom's exact coordinates
# Use Google Maps to find your classroom's Lat/Lon
CLASSROOM_LAT = 15.626 
CLASSROOM_LON = 76.897
ALLOWED_RADIUS = 0.02 # Roughly 2km for testing. Change to 0.001 for classroom-only.

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="AI College Portal", page_icon="🎓", layout="wide")

st.sidebar.title("🏫 Navigation")
page = st.sidebar.radio("Go to:", ["Attendance (Individual Login)", "New User Registration", "Public Results"])

# --- HELPERS ---
def check_geofence(loc):
    if not loc: return False
    lat_diff = abs(loc['coords']['latitude'] - CLASSROOM_LAT)
    lon_diff = abs(loc['coords']['longitude'] - CLASSROOM_LON)
    return lat_diff < ALLOWED_RADIUS and lon_diff < ALLOWED_RADIUS

# --- PAGE 1: NEW USER REGISTRATION ---
if page == "New User Registration":
    st.header("📝 Student Self-Registration")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name")
        usn = st.text_input("USN (Unique Student Number)")
    
    st.write("### Profile Photo")
    st.caption("This photo will be used for AI facial recognition during attendance.")
    reg_photo = st.camera_input("Capture Profile")
    
    if st.button("Submit Registration"):
        if full_name and usn and reg_photo:
            with st.spinner("Processing Registration..."):
                img_data = reg_photo.getvalue()
                
                # 1. Store photo in S3
                s3.put_object(Bucket=BUCKET_NAME, Key=f"profiles/{usn}.jpg", Body=img_data)
                
                # 2. Add Face to Rekognition Collection
                rekog.index_faces(
                    CollectionId=COLLECTION_ID,
                    Image={'Bytes': img_data},
                    ExternalImageId=usn,
                    MaxFaces=1
                )
                
                # 3. Save Info to DynamoDB
                dynamo.Table(TABLE_PROFILES).put_item(Item={'USN': usn, 'Name': full_name})
                
                st.success(f"Successfully registered {full_name} ({usn})!")
        else:
            st.warning("Please fill in all fields and take a photo.")

# --- PAGE 2: ATTENDANCE (LOGIN) ---
elif page == "Attendance (Individual Login)":
    st.header("📸 AI Attendance Kiosk")
    st.markdown("---")
    
    # 1. Check Geolocation first
    st.write("### Step 1: Location Verification")
    user_location = get_geolocation()
    
    if user_location:
        if check_geofence(user_location):
            st.success("📍 Location Verified. You are in the authorized classroom area.")
            
            # 2. Proceed to Face Auth
            st.write("### Step 2: Facial Recognition")
            login_photo = st.camera_input("Look at the camera")
            
            if login_photo:
                with st.spinner("Verifying Identity..."):
                    login_bytes = login_photo.getvalue()
                    try:
                        response = rekog.search_faces_by_image(
                            CollectionId=COLLECTION_ID,
                            Image={'Bytes': login_bytes},
                            MaxFaces=1,
                            FaceMatchThreshold=90
                        )
                        
                        if response['FaceMatches']:
                            recognized_usn = response['FaceMatches'][0]['Face']['ExternalImageId']
                            st.balloons()
                            st.success(f"Verified! Attendance marked for USN: {recognized_usn}")
                            
                            # 3. Log Attendance in DynamoDB
                            dynamo.Table(TABLE_ATTENDANCE).put_item(Item={
                                'USN': recognized_usn,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Status': 'Present'
                            })
                        else:
                            st.error("Face not recognized. Please Register as a New User.")
                    except Exception:
                        st.error("Error detecting face. Ensure you are in good lighting.")
        else:
            st.error("🚫 Access Denied: You must be physically present in the classroom.")
            st.info(f"Geofence Required: Lat {CLASSROOM_LAT}, Lon {CLASSROOM_LON}")
    else:
        st.warning("Please enable GPS/Location access in your browser to verify your presence.")

# --- PAGE 3: PUBLIC RESULTS ---
elif page == "Public Results":
    st.header("📊 Student Performance Board")
    st.info("Publicly available results for the entire batch.")
    st.markdown("---")
    
    try:
        # Scan Results Table
        table = dynamo.Table(TABLE_RESULTS)
        items = table.scan()['Items']
        
        if items:
            results_df = pd.DataFrame(items)
            # Reorder columns to show USN first if it exists
            if 'USN' in results_df.columns:
                cols = ['USN'] + [c for c in results_df.columns if c != 'USN']
                results_df = results_df[cols]
            
            st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("No results have been uploaded to the database yet.")
    except Exception:
        st.error("Database connection error. Check if the 'StudentResults' table exists.")
