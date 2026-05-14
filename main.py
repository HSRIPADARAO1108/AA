import streamlit as st
import boto3
import pandas as pd
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
    #     return False
    # lat_diff = abs(loc['coords']['latitude'] - CLASSROOM_LAT)
    # lon_diff = abs(loc['coords']['longitude'] - CLASSROOM_LON)
    # return lat_diff < ALLOWED_RADIUS and lon_diff < ALLOWED_RADIUS

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
    
    # 1. Check Location
    user_loc = get_geolocation()
    
    if user_loc:
        if check_location(user_loc):
            st.success("Access Granted: Location Verified.")
            
            # 2. Face Recognition
            login_photo = st.camera_input("Look at the camera to log in")
            
            if login_photo:
                with st.spinner("Matching Face..."):
                    login_bytes = login_photo.getvalue()
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
                        else:
                            st.error("Identity not verified. Please ensure you are registered.")
                    except Exception as e:
                        st.error("Verification failed. Please check your lighting and camera.")
        else:
            st.error("🚫 Access Denied: You are not in the classroom.")
    else:
        st.warning("Please allow browser location access to continue.")

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
