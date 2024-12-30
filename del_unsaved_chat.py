def delete_unsaved_chats():
    """Delete chat sessions without summaries and their associated messages."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Find all session IDs without summaries
        cursor.execute("SELECT id FROM chat_sessions WHERE summary IS NULL")
        unsaved_session_ids = [row[0] for row in cursor.fetchall()]

        if unsaved_session_ids:
            # Delete corresponding messages
            cursor.execute(
                "DELETE FROM messages WHERE session_id IN ({})".format(
                    ",".join("?" * len(unsaved_session_ids))
                ),
                unsaved_session_ids,
            )

            # Delete the unsaved sessions
            cursor.execute(
                "DELETE FROM chat_sessions WHERE id IN ({})".format(
                    ",".join("?" * len(unsaved_session_ids))
                ),
                unsaved_session_ids,
            )

        conn.commit()
