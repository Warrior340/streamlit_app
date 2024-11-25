import streamlit as st
import sqlite3
import json
import os
from datetime import datetime, timedelta
from hashlib import sha256

# Initialize database for chats
def init_chat_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            chat TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Save chat to the database
def save_chat(username, chat):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (username, chat) VALUES (?, ?)", (username, chat))
    
    # Delete chats older than 5 days
    five_days_ago = datetime.now() - timedelta(days=5)
    c.execute("DELETE FROM chat_history WHERE timestamp < ?", (five_days_ago,))
    
    conn.commit()
    conn.close()

# Retrieve chats from the last 5 days for a user
def get_last_5_days_chats(username):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    five_days_ago = datetime.now() - timedelta(days=5)
    c.execute("""
        SELECT chat, timestamp FROM chat_history
        WHERE username = ? AND timestamp >= ?
        ORDER BY timestamp DESC
    """, (username, five_days_ago))
    chats = c.fetchall()
    conn.close()
    return chats

# User management functions
USER_FILE = "users.json"

def init_user_file():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)

def hash_password(password):
    return sha256(password.encode()).hexdigest()

def register_user(username, password):
    with open(USER_FILE, "r") as f:
        users = json.load(f)

    if username in users:
        return False  # User already exists

    users[username] = hash_password(password)
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

    return True

def authenticate_user(username, password):
    with open(USER_FILE, "r") as f:
        users = json.load(f)

    if username in users and users[username] == hash_password(password):
        return True
    return False

# Initialize files
init_chat_db()
init_user_file()

# Main Streamlit App
st.set_page_config(page_title="AI Chat Application", layout="centered")
st.title("AI Chat Application")

# Navigation
page = st.sidebar.selectbox("Navigation", ["Login", "Register", "AI Chat", "Chat History"])

if page == "Login":
    st.subheader("Login")
    login_username = st.text_input("Username", key="login_username")
    login_password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Login")

    if login_button:
        if login_username and login_password:
            if authenticate_user(login_username, login_password):
                st.session_state["username"] = login_username
                st.success(f"Welcome, {login_username}! Navigate to AI Chat or Chat History.")
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Please fill in all fields.")

elif page == "Register":
    st.subheader("Register")
    reg_username = st.text_input("Create Username", key="reg_username")
    reg_password = st.text_input("Create Password", type="password", key="reg_password")
    reg_button = st.button("Register")

    if reg_button:
        if reg_username and reg_password:
            if register_user(reg_username, reg_password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Try a different one.")
        else:
            st.error("Please fill in all fields.")

elif page == "AI Chat":
    if "username" in st.session_state:
        st.subheader("Chat with AI")
        chat_input = st.text_input("Enter your message", "")
        send_button = st.button("Send")

        if send_button:
            if chat_input:
                st.write(f"You: {chat_input}")
                response = f"AI: Echoing '{chat_input}'"  # Replace with AI model logic
                st.write(response)
                save_chat(st.session_state["username"], f"You: {chat_input}")
                save_chat(st.session_state["username"], response)
    else:
        st.warning("Please log in to access the chat.")

elif page == "Chat History":
    if "username" in st.session_state:
        st.subheader("Chat History (Last 5 Days)")
        chats = get_last_5_days_chats(st.session_state["username"])
        if chats:
            for chat, timestamp in chats:
                st.write(f"- {chat} ({timestamp.split('.')[0]})")
        else:
            st.write("No chat history found.")
    else:
        st.warning("Please log in to view your chat history.")
