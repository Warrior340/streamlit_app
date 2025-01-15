import streamlit as st
import sqlite3
from datetime import datetime, timedelta
from threading import Lock
from http.cookies import SimpleCookie
import os


# Helper function to set cookies
def set_cookie(key, value):
    cookie = SimpleCookie()
    cookie[key] = value
    cookie[key]["path"] = "/"
    os.environ["HTTP_COOKIE"] = cookie.output(header="", sep="")

# Helper function to get cookies
def get_cookie(key):
    cookie_header = os.environ.get("HTTP_COOKIE", "")
    cookies = SimpleCookie(cookie_header)
    return cookies.get(key).value if key in cookies else None

# Helper function to delete cookies
def delete_cookie(key):
    set_cookie(key, "")


# Database setup
db_lock = Lock()

def get_db_connection():
    """Create a new database connection for each thread."""
    return sqlite3.connect("chat_app.db")

# Initialize database
def initialize_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_sessions WHERE start_time < ?", (expiry_date,))
        conn.commit()

def register_user(username, password):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def login_user(username, password):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
        return cursor.fetchone()

def create_new_session(user_id=None):
    start_time = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, start_time, summary) VALUES (?, ?, ?)",
            (user_id, start_time, None)
        )
        conn.commit()
        return cursor.lastrowid

def get_saved_sessions(user_id):
    """Fetch saved chat sessions with user-provided summaries, grouped by date."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, start_time, summary
            FROM chat_sessions
            WHERE user_id = ? AND summary IS NOT NULL
            ORDER BY start_time DESC
        """, (user_id,))
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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, sender, message, timestamp)
        )
        conn.commit()

def get_messages(session_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sender, message, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        messages = cursor.fetchall()
        return [(sender, message, datetime.fromisoformat(timestamp)) for sender, message, timestamp in messages]

def save_chat_session(session_id, summary):
    with db_lock, get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_sessions
            SET summary = ?
            WHERE id = ?
        """, (summary, session_id))
        conn.commit()

# Streamlit dialog for saving chat
@st.dialog("Save Chat Summary")
def save_chat():
    st.write("Provide a summary for this chat session:")
    summary = st.text_input("Chat Summary", key="chat_summary")
    if st.button("Save"):
        if summary:
            save_chat_session(st.session_state.current_session, summary)
            st.success("Chat summary saved successfully!")
            st.rerun()
        else:
            st.error("Summary cannot be empty.")

# Session state initialization
if "logged_in" not in st.session_state:
    user_id = get_cookie("user_id")
   
    print(f"the id is {user_id}")
    #print(username)
    if user_id :
        
        print(user_id.split())
        st.session_state.logged_in = True
        st.session_state.user_id = int(user_id.split()[1])  # Fetch user ID from DB
        st.session_state.username=  user_id.split()[0]
        st.session_state.current_session = create_new_session(st.session_state.user_id)
        st.session_state.chat_history = []
        st.session_state.page = "Chat Assistant"
        
    else:       
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_session = None
        st.session_state.chat_history = []
        st.session_state.page = "Login"

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.current_session = None
    st.session_state.chat_history = []
    st.session_state.page = "Login"
    delete_cookie("user_id")
    
    st.rerun()

# Chat assistant page
def chat_assistant_page():
    with st.sidebar:
        if st.button("Logout"):
            logout()

    st.title("Chat Assistant")
    st.subheader(f"Welcome, {st.session_state.username or 'Guest'}")

    # Save Chat functionality
    with st.sidebar:
        if st.button("Save Chat"):
            save_chat()

    # Display saved sessions
    if st.session_state.logged_in:
        saved_sessions_by_date = get_saved_sessions(st.session_state.user_id)
        st.sidebar.header("Saved Chats")
        for date, sessions in sorted(saved_sessions_by_date.items(), reverse=True):
            st.sidebar.subheader(str(date))
            for session_id, start_time, summary in sessions:
                if st.sidebar.button(f"{summary} ({start_time[11:16]})", key=f"saved_session_{session_id}"):
                    st.session_state.current_session = session_id
                    st.session_state.chat_history = get_messages(session_id)
                    st.rerun()

    # Chat input and history
    user_input = st.chat_input("Type your message...")
    if user_input:
        save_message(st.session_state.current_session, "You", user_input)
        response = f"Response to: {user_input}"  # Placeholder for AI response
        save_message(st.session_state.current_session, "Assistant", response)
        st.session_state.chat_history = get_messages(st.session_state.current_session)
        st.rerun()

    # Display chat history
    st.write("### Chat History")
    for sender, message, timestamp in st.session_state.chat_history:
        st.write(f"**{sender}**: {message} ({timestamp.strftime('%H:%M')})")

# Authentication pages
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

    # Add a Back to Login button
    if st.button("Back to Login"):
        st.session_state.page = "Login"
        st.rerun()

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
            print(f"id in login {st.session_state.user_id}")
            st.session_state.current_session = create_new_session(user[0])
            cook=f"{st.session_state.username} {str(st.session_state.user_id)}"
            set_cookie("user_id", cook )
           
            print("-------------------------")
            print(get_cookie("user_id"))
           
            st.session_state.page = "Chat Assistant"
            st.rerun()
        else:
            st.error("Invalid username or password.")

    if st.button("Continue as Guest"):
        st.session_state.page = "Chat Assistant"
        st.session_state.current_session = create_new_session()
        st.rerun()

    # Add a Register button for navigation
    if st.button("Register"):
        st.session_state.page = "Register"
        st.rerun()

# Initialize database
initialize_database()

# Page navigation
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Register":
    register_page()
elif st.session_state.page == "Chat Assistant":
    chat_assistant_page()
