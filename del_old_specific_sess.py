def delete_old_or_specific_session(session_id=None, user_id=None):
    """
    Deletes chat sessions without summaries older than 1 day or a specific session,
    along with all corresponding messages.
    """
    expiry_date = (datetime.now() - timedelta(days=1)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Step 1: Delete a specific session and its messages
        if session_id:
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ? AND summary IS NULL", (session_id,))

        # Step 2: Delete sessions older than 1 day and their messages
        elif user_id is not None:
            cursor.execute("""
                SELECT id FROM chat_sessions
                WHERE user_id = ? AND summary IS NULL AND start_time < ?
            """, (user_id, expiry_date))
            unsaved_session_ids = [row[0] for row in cursor.fetchall()]

            for unsaved_session_id in unsaved_session_ids:
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (unsaved_session_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (unsaved_session_id,))
        else:
            cursor.execute("""
                SELECT id FROM chat_sessions
                WHERE user_id IS NULL AND summary IS NULL AND start_time < ?
            """, (expiry_date,))
            unsaved_session_ids = [row[0] for row in cursor.fetchall()]

            for unsaved_session_id in unsaved_session_ids:
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (unsaved_session_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (unsaved_session_id,))

        conn.commit()
