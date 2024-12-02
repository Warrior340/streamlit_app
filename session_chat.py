import streamlit as st
import sqlite3
from datetime import datetime, timedelta

# Database setup
conn = sqlite3.connect("chat_app.db")
cursor = conn.cursor()

# Create tables if not exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    start_time DATETIME,
    summary TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    sender TEXT,
    message TEXT,
    timestamp DATETIME,
    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
)
""")
conn.commit()

# Helper functions
def delete_old_sessions():
    """Delete sessions older than 7 days."""
    expiry_date = (datetime.now() - timedelta(days=7)).isoformat()
    cursor.execute("DELETE FROM chat_sessions WHERE start_time < ?", (expiry_date,))
    conn.commit()

def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone()

def create_new_session(user_id=None):
    start_time = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO chat_sessions (user_id, start_time, summary) VALUES (?, ?, ?)",
        (user_id, start_time, "New session started") if user_id else (None, start_time, "Guest session")
    )
    conn.commit()
    return cursor.lastrowid

def get_sessions(user_id):
    """Fetch all sessions within the last 7 days, grouped by date."""
    seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
    cursor.execute("""
        SELECT id, start_time, summary
        FROM chat_sessions
        WHERE user_id = ? AND start_time >= ?
        ORDER BY start_time DESC
    """, (user_id, seven_days_ago))
    sessions = cursor.fetchall()
    grouped_sessions = {}
    for session_id, start_time, summary in sessions:
        date = datetime.fromisoformat(start_time).date()
        if date not in grouped_sessions:
            grouped_sessions[date] = []
        grouped_sessions[date].append((session_id, start_time, summary))
    return grouped_sessions

def save_message(session_id, sender, message):
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO messages (session_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, sender, message, timestamp)
    )
    conn.commit()

def get_messages(session_id):
    cursor.execute("""
        SELECT sender, message, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = cursor.fetchall()
    return [(sender, message, datetime.fromisoformat(timestamp)) for sender, message, timestamp in messages]

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.current_session = None
    st.session_state.chat_history = []
    st.session_state.page = "Login"  # Default page
    st.session_state.is_guest = False  # To differentiate guest and logged-in users

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.current_session = None
    st.session_state.chat_history = []
    st.session_state.page = "Login"
    st.session_state.is_guest = False
    st.rerun()

# User authentication
def register_page():
    st.subheader("Register")
    username = st.text_input("Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registration successful! Redirecting to login...")
            st.session_state.page = "Login"
            st.rerun()
        else:
            st.error("Username already exists.")

def login_page():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.username = username
            st.session_state.current_session = create_new_session(user[0])  # New session created on login
            st.session_state.page = "Chat Assistant"
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    
    # Add Register button
    if st.button("Register"):
        st.session_state.page = "Register"  # Redirect to registration page
        st.rerun()
    
    # Add "Continue Without Login" button
    if st.button("Continue Without Login"):
        st.session_state.is_guest = True
        st.session_state.page = "Chat Assistant"
        st.session_state.current_session = create_new_session()  # Guest session
        st.rerun()

# Chat assistant
def chat_assistant_page():
    # Logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            logout()
    
    st.title("Chat Assistant")
    if st.session_state.is_guest:
        st.subheader("Guest Mode")
    else:
        st.subheader(f"Welcome {st.session_state.username}")

    # Delete old sessions
    delete_old_sessions()

    # Display sessions for logged-in users
    if not st.session_state.is_guest:
        sessions_by_date = get_sessions(st.session_state.user_id)
        st.sidebar.header("Chat Sessions (Last 7 Days)")
        for date, sessions in sorted(sessions_by_date.items(), reverse=True):
            st.sidebar.subheader(str(date))
            for session_id, start_time, summary in sessions:
                # Get messages for this session
                messages = get_messages(session_id)
                if messages:  # Only show non-empty sessions
                    time = datetime.fromisoformat(start_time).strftime('%H:%M')
                    if st.sidebar.button(f"{time} - {summary}", key=f"session_{session_id}"):
                        st.session_state.current_session = session_id
                        st.session_state.chat_history = messages
                        st.rerun()

    # New session button
    if not st.session_state.is_guest and st.button("Start New Session"):
        st.session_state.current_session = create_new_session(st.session_state.user_id)
        st.session_state.chat_history = []
        st.rerun()

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        save_message(st.session_state.current_session, "You", user_input)
        response = f"Response to: {user_input}"
        save_message(st.session_state.current_session, "Assistant", response)
        cursor.execute("""
            UPDATE chat_sessions
            SET summary = ?
            WHERE id = ?
        """, (response[:30], st.session_state.current_session))
        conn.commit()
        st.session_state.chat_history = get_messages(st.session_state.current_session)
        st.rerun()

    # Display chat history
    st.write("### Chat History")
    for sender, message, timestamp in st.session_state.chat_history:
        st.write(f"**{sender}**: {message} ({timestamp.strftime('%H:%M')})")

# Page navigation
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Register":
    register_page()
elif st.session_state.page == "Chat Assistant":
    chat_assistant_page()
