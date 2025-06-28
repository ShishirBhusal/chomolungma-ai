# app_ui.py (V3.9 - Final Definitive Version)

import streamlit as st
import requests
import base64
from io import BytesIO
import time

# --- 1. API Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
LOGIN_URL = f"{API_BASE_URL}/auth/token"
REGISTER_URL = f"{API_BASE_URL}/auth/register"
AGENT_URL = f"{API_BASE_URL}/agent/invoke"

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Chomolungma - AI Trekking Guide",
    page_icon="🏔️",
    layout="wide"
)

# --- 3. Session State Management ---
def initialize_state():
    defaults = {
        "logged_in": False,
        "auth_token": None,
        "messages": [],
        "page": "login",
        "staged_image": None,
        "uploader_key": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_state()

# --- 4. Authentication Functions ---
def handle_login(email, password):
    try:
        login_data = {'username': email, 'password': password}
        response = requests.post(LOGIN_URL, data=login_data)
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.logged_in = True
            st.session_state.auth_token = token_data['access_token']
            st.session_state.messages = []
            st.rerun()
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the auth service: {e}")

def handle_register(email, password):
    try:
        register_data = {'email': email, 'password': password}
        response = requests.post(REGISTER_URL, json=register_data)
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the registration service: {e}")

# --- 5. UI Rendering ---

# Login/Register Page
if not st.session_state.logged_in:
    st.title("Welcome to Chomolungma AI 🏔️")
    if st.session_state.page == "login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                handle_login(email, password)
        if st.button("Register Here", key="to_register"):
            st.session_state.page = "register"; st.rerun()
    elif st.session_state.page == "register":
        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if password and password == confirm_password:
                    handle_register(email, password)
                else:
                    st.error("Passwords do not match or are empty.")
        if st.button("Back to Login", key="to_login"):
            st.session_state.page = "login"; st.rerun()

# Main Chat Interface
else:
    st.sidebar.title("Chomolungma AI")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            if key != 'page': del st.session_state[key]
        initialize_state(); st.session_state.page = "login"; st.rerun()

    st.title("Chomolungma - Your AI Trekking Guide 🏔️")

    # Display initial welcome message if chat is empty
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Namaste! I am Chomolungma. How can I help you plan your trek today?"})

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part["type"] == "text": st.markdown(part["text"])
                    elif part["type"] == "image_url":
                        img_data = base64.b64decode(part["image_url"].split(",", 1)[1])
                        st.image(img_data, width=250)
            else:
                st.markdown(msg["content"])

    # Image uploader and staging logic
    uploaded_file = st.file_uploader(
        "Attach an image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        base64_str = base64.b64encode(bytes_data).decode()
        st.session_state.staged_image = {
            "mime_type": uploaded_file.type,
            "data": base64_str,
            "preview": bytes_data # Store raw bytes for preview
        }

    # Display the staged image if it exists
    if st.session_state.staged_image:
        st.image(
            st.session_state.staged_image["preview"],
            caption="Image staged for the next message.",
            width=200
        )

    # Chat input and processing logic
    if prompt := st.chat_input("Ask me to plan your trek..."):
        # 1. Grab the staged image payload for the API call
        image_payload = st.session_state.staged_image

        # 2. Construct the user message for display in the chat history
        user_message_for_display = [{"type": "text", "text": prompt}]
        if image_payload:
            img_url = f"data:{image_payload['mime_type']};base64,{image_payload['data']}"
            user_message_for_display.append({"type": "image_url", "image_url": img_url})
        st.session_state.messages.append({"role": "user", "content": user_message_for_display})

        # Display the user message immediately
        with st.chat_message("user"):
            for part in user_message_for_display:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    img_data = base64.b64decode(part["image_url"].split(",", 1)[1])
                    st.image(img_data, width=250)

        # 3. Call the API and display the response
        # --- Streaming request to backend ---
        try:
            # Prepare payload for the API (only use mime_type and data)
            api_image_payload = {
                "mime_type": image_payload["mime_type"],
                "data": image_payload["data"]
            } if image_payload else None

            api_payload = {"query": prompt, "image": api_image_payload}
            headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

            # Open the assistant chat bubble immediately
            with st.chat_message("assistant"):
                status_box = st.empty()
                content_box = st.empty()
                # Buffer to hold the growing status log
                status_log = ""

                # Show a progress spinner that updates based on time elapsed
                spinner_placeholder = st.empty()
                start_time = time.time()
                
                def get_spinner_message(elapsed):
                    if elapsed > 15:  # After 15 seconds
                        return "⏳ This is taking longer than expected. Almost there..."
                    elif elapsed > 10:  # After 10 seconds
                        return "🔄 Finalizing the details..."
                    elif elapsed > 5:   # After 5 seconds
                        return "🔄 Gathering more information..."
                    else:
                        return "🔄 Thinking..."
                
                # Initial spinner
                spinner_placeholder.info(get_spinner_message(0))
                
                try:
                    response = requests.post(AGENT_URL, json=api_payload, headers=headers, stream=True)
                    accumulated_answer = ""
                    
                    # Check and update spinner every second
                    last_update = time.time()
                    update_interval = 1.0  # Update spinner every 1 second

                    for chunk in response.iter_lines(decode_unicode=True):
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        # Update spinner message if needed
                        if current_time - last_update >= update_interval:
                            spinner_placeholder.info(get_spinner_message(elapsed))
                            last_update = current_time
                            
                        if not chunk:
                            continue
                            
                        # --- Handle STATUS updates ---
                        if chunk.startswith("STATUS:"):
                            status_line = chunk[len("STATUS:"):]
                            # Preserve incoming newlines (they may be encoded as \n)
                            status_line = status_line.replace("\\n", "\n")
                            status_log += status_line + "\n"
                            # Render as a code block so it looks like a console trace
                            status_box.code(status_log, language="")
                        # --- Handle streamed CONTENT tokens ---
                        elif chunk.startswith("CONTENT:"):
                            text_frag = chunk[len("CONTENT:"):]
                            text_frag_decoded = text_frag.replace("\\n", "\n")
                            accumulated_answer += text_frag_decoded
                            content_box.markdown(accumulated_answer, unsafe_allow_html=True)
                    
                    # Clear the spinner when done
                    spinner_placeholder.empty()
                    
                except Exception as e:
                    spinner_placeholder.empty()
                    raise e

            # Append assistant's response to history
            st.session_state.messages.append({"role": "assistant", "content": accumulated_answer})

        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # 4. Clear the staged image and reset the uploader for the next turn
        st.session_state.staged_image = None
        st.session_state.uploader_key += 1
        
        # 5. Rerun the script to display the new messages and clear the preview
        st.rerun()