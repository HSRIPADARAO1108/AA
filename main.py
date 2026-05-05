import streamlit as st
import boto3
import pandas as pd
from datetime import datetime
from streamlit_js_eval import get_geolocation # For Geofencing

# --- CONFIG ---
CLASSROOM_LAT = 12.9716  # Replace with your classroom's Latitude
CLASSROOM_LON = 77.5946  # Replace with your classroom's Longitude
ALLOWED_DISTANCE = 0.05   # Accuracy radius (~50 meters)

rekog = boto3.client('rekognition', region_name='us-east-1')
dynamo = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3')

# --- NAVIGATION ---
st.sidebar.title("College Portal")
page = st.sidebar.radio("Go to", ["Login & Attendance", "New User Registration", "Public Results"])

# --- HELPERS ---
def is_inside_class(loc):
    if not loc: return False
    lat_dist = abs(loc['coords']['latitude'] - CLASSROOM_LAT)
    lon_dist = abs(loc['coords']['longitude'] - CLASSROOM_LON)
    return lat_dist < ALLOWED_DISTANCE and lon_dist < ALLOWED_DISTANCE

# --- 1. NEW USER REGISTRATION ---
if page == "New User Registration":
    st.header("📝 Register New Student")
    name = st.text_input("Full Name")
    usn = st.text_input("USN (e.g., 1DA25SCS01)")
    photo = st.camera_input("Take Official Profile Photo")
    
    if st.button("Complete Registration"):
        if name and usn and photo:
            img_bytes = photo.getvalue()
            # Save to S3 and Index Face
            s3.put_object(Bucket='college-system-data', Key=f"profile_pics/{usn}.jpg", Body=img_bytes)
            rekog.index_faces(CollectionId="college_faces", Image={'Bytes': img_bytes}, ExternalImageId=usn)
            # Save Profile Info
            dynamo.Table('StudentProfiles').put_item(Item={'USN': usn, 'Name': name})
            st.success(f"Student {usn} registered successfully!")

# --- 2. LOGIN & ATTENDANCE (WITH GEOFENCING) ---
elif page == "Login & Attendance":
    st.header("📸 Student Attendance (In-Class Only)")
    
    # Check Location FIRST
    loc = get_geolocation()
    
    if loc:
        if is_inside_class(loc):
            st.success("📍 Location Verified: You are in the classroom.")
            login_photo = st.camera_input("Scan Face to Login")
            
            if login_photo:
                img_bytes = login_photo.getvalue()
                try:
                    res = rekog.search_faces_by_image(CollectionId="college_faces", Image={'Bytes': img_bytes}, MaxFaces=1)
                    if res['FaceMatches']:
                        usn = res['FaceMatches'][0]['Face']['ExternalImageId']
                        st.balloons()
                        st.success(f"Attendance Marked for USN: {usn}")
                        
                        # Log Attendance
                        dynamo.Table('AttendanceLogs').put_item(Item={
                            'USN': usn,
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Status': 'Present',
                            'Location': f"{loc['coords']['latitude']}, {loc['coords']['longitude']}"
                        })
                    else:
                        st.error("Face not recognized!")
                except:
                    st.error("Identification failed. Please try again.")
        else:
            st.error("🚫 Access Denied: You must be physically present in the classroom to take attendance.")
    else:
        st.warning("Please allow location access to continue.")

# --- 3. PUBLIC RESULTS (OPEN ACCESS) ---
elif page == "Public Results":
    st.header("📊 Batch Results (Public)")
    # Access for anyone - no login required
    results_table = dynamo.Table('StudentResults').scan()
    if results_table['Items']:
        df = pd.DataFrame(results_table['Items'])
        st.dataframe(df.sort_values(by='USN'))
    else:
        st.info("No results published yet.")
