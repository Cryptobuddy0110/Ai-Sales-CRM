import streamlit as st
import requests
import time
import platform
import json
import psutil
from database import add_feedback, store_user_feedback_metadata
from streamlit.components.v1 import html as components_html

st.set_page_config(page_title="Project Feedback Form", page_icon="📝")

# ✅ Get project name from URL parameters
query_params = st.query_params
project_name = query_params.get("project", "")

if not project_name:
    st.error("Invalid feedback form. No project selected.")
    st.stop()



# ✅ Capture Browser Info & Referrer URL
user_agent = st.query_params.get("user-agent", ["Unknown"])[0]
referrer = st.query_params.get("referrer", ["Direct Access"])[0]

# ✅ Try to Get User's IP Address & Approximate Location
try:
    ip_info = requests.get("https://ipinfo.io/json").json()
    ip_address = ip_info.get("ip", "Unknown")
    location = ip_info.get("loc", "Unknown")  # Returns lat,long if available
except:
    ip_address = "Unknown"
    location = "Unknown"

# ✅ Capture Device Info
device_info = {
    "OS": platform.system(),
    "OS Version": platform.release(),
    "Machine": platform.machine(),
    "Processor": platform.processor()
}

# ✅ Capture Battery Info (Limited in Web Browsers)
try:
    battery = psutil.sensors_battery()
    battery_percentage = battery.percent if battery else "Unknown"
except:
    battery_percentage = "Unknown"

# ✅ Initialize session state to track metadata submission
if "metadata_submitted" not in st.session_state:
    st.session_state["metadata_submitted"] = False

# ✅ Store Data in DB only if it hasn't been stored in this session
if not st.session_state["metadata_submitted"]:
    store_user_feedback_metadata(
        project_name=project_name,
        ip_address=ip_address,
        location=location,
        browser_info=user_agent,
        referrer_url=referrer,
        device_info=json.dumps(device_info),
        battery_life=battery_percentage
    )
    st.session_state["metadata_submitted"] = True  # Mark as submitted

# ✅ Create a placeholder for dynamic content
def display_html_page():
    with open("sub.html", "r") as file:
        html_content = file.read()
    components_html(html_content,height=500,scrolling=False)
    
placeholder = st.empty()


# ✅ Feedback Form
with placeholder.container():
    with st.container(border=True):
            st.subheader(f"Feedback for Project: {project_name}")
            client_name = st.text_input("Your Name", placeholder="Enter Your Name", disabled=False, key="name")
            rating = st.feedback("stars", key="rating")
            comments = st.text_area("Your Feedback", placeholder="Enter your feedback here...", disabled=False, key="comments")
            submit_button = st.button("Submit Feedback", key="submit")
            
  
    
    if submit_button:
      
        if client_name:
            
            add_feedback(project_name, client_name, rating, comments)
            with placeholder.container():
                display_html_page()

            
          
            # placeholder.empty()  # Clear the container
            st.toast("Feedback Submitted!", icon="✅")
     
   

        else:
            st.warning("Please enter your name.")
    