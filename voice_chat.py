from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
import os
from pydub import AudioSegment

load_dotenv()

pool = ConnectionPool(conninfo=os.getenv('POSTGRES_SUPABASE')) 

def insert_file(from_name: str,to_name: str, from_uid: str, to_uid: str, filename: str, read: bool = False, ftype: str = "voice") -> int | None:

    new_id = None

    # Acquire a connection from the pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.prepare_threshold = None
                # Insert the row; `id` is bigserial, `created_at` defaults to now()
                cur.execute(
                    """
                    INSERT INTO voice_chat (from_uid, from_name,to_name,to_uid, filename, read, file_type)
                    VALUES (%s, %s,%s,%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (from_uid, from_name,to_name,to_uid, filename, read,ftype),
                )
                new_id = cur.fetchone()[0]
                # When exiting the `with conn:` block, psycopg3 will automatically commit 
                # (because autocommit=False). If an exception occurs, it will rollback.
                print(f"__Inserted voice record with id = {new_id}")
            except Exception as e:
                conn.rollback() 
                # Any exception here will cause a rollback at context exit
                print(f"__Error during INSERT: {e}")
                return {"Error": str(e)}
                new_id = None

    return {"id":new_id}

def get_unread_by_to_uid(to_uid: str) -> list[dict]:
    """
    Returns a list of dicts, each representing one unread voice record
    for the specified `to_uid`.
    """
    
    results = []

    with pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT id, from_uid, from_name,filename, created_at
                    FROM voice_chat
                    WHERE to_uid = %s
                      AND read = FALSE AND file_type = 'voice'
                    ORDER BY created_at desc;
                    """,
                    (to_uid,),
                )
                rows = cur.fetchall()
                for row in rows:
                    # row = (id, from_uid, filename, created_at)
                    record = {
                        "id": row[0],
                        "from_uid": row[1],
                        "from_name": row[2],
                        "filename": row[3],
                        "created_at": row[4],
                    }
                    results.append(record)
            except Exception as e:
                print(f"[!] Error during SELECT: {e}")
                # No rollback needed for a SELECT, but it will “exit” the transaction block if used.
    return results

def get_chat_history(to_uid: str, from_uid: str) -> list[dict]:
    """
    Returns a list of dicts, each representing one unread voice record
    for the specified `to_uid`.
    """
    
    results = []
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT id, from_uid, from_name,to_uid,filename, created_at, file_type
                        FROM voice_chat
                        WHERE (to_uid = %s
                        AND from_uid = %s) OR (from_uid = %s AND to_uid = %s)
                        ORDER BY created_at desc;
                        """,
                        (to_uid, from_uid,to_uid, from_uid),
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        # row = (id, from_uid, filename, created_at)
                        record = {
                            "id": str(row[0]),
                            "from_uid": row[1],
                            "from_name": row[2],
                            "to_uid": row[3],
                            "filename": row[4],
                            "created_at": str(row[5]),
                            "type": row[6],
                        }
                        print(record)
                        results.append(record)
                except Exception as e:
                    print(f"[!] Error during SELECT: {e}")
                    # No rollback needed for a SELECT, but it will “exit” the transaction block if used.
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
    
    return results


def mark_read(id):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE voice_chat
                    SET read = TRUE
                    WHERE id = %s;
                    """,
                    (id,),
                )
    except Exception as e:
        print(f"[!] Error during UPDATE: {e}")


def get_chat_list(to_uid):
    results = []
    try:
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, from_uid, to_uid, from_name, to_name, created_at
                    FROM voice_chat
                    WHERE from_uid = %s OR to_uid = %s
                    ORDER BY created_at DESC
                """, (to_uid, to_uid))

                seen = {}
                for id_, f, t, fname, tname, ct in cur:
                    # figure out who the “other” user is
                    other_uid = t if f == to_uid else f
                    if other_uid not in seen:
                        # pick the correct name based on direction
                        other_name = tname if f == to_uid else fname
                        if other_name is None:
                            other_name = ""
                        seen[other_uid] = {
                            "id":          id_,
                            "uid":         other_uid,
                            "name":        other_name if not None else "",
                            "my_name":     fname if f == to_uid else tname,
                            "created_at":  ct,
                            "to_uid": to_uid
                        }

                return list(seen.values())

    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
        return []
import speech_recognition as sr
UPLOAD_DIR_CHAT = os.path.join(os.path.dirname(__file__), 'uploads_chat')
def get_transcripted_message_by_id(id):
    tmp_path = None
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                print("[DEBUG] Executing SELECT query for message ID:", id)
                cur.execute(
                    """
                    SELECT filename
                    FROM voice_chat
                    WHERE id = %s
                    LIMIT 1;
                    """,
                    (id,)
                )
                row = cur.fetchone()
                if row:
                    tmp_path = row[0]
                    print("[DEBUG] Found filename:", tmp_path)
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
    
    if tmp_path is None:
        print("[DEBUG] No message found for ID:", id)
        return "No message found"
    
    r = sr.Recognizer()

    try:
        # Transcript
        tmp_full = os.path.join(UPLOAD_DIR_CHAT, tmp_path)
        print("[DEBUG] Full path to file:", tmp_full)
        
        # Load and convert audio
        fixed_path = "fixed.wav"
        print("[DEBUG] Converting audio to fixed format:", fixed_path)

        audio_seg = AudioSegment.from_file(tmp_full)
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
        audio_seg.export(fixed_path, format="wav", codec="pcm_s16le")

        # Load into speech_recognition
        r = sr.Recognizer()
        with sr.AudioFile(fixed_path) as source:
            print("[DEBUG] Recording audio from fixed path")
            audio_data = r.record(source)      

        text = r.recognize_openai(
            audio_data,
            model="whisper-1"
        )
        print("[DEBUG] Transcription result:", text)
        return text
    except Exception as e:
        print('[!] Error during transcription:', e)
        return "Internal Server Error"

def get_transcripted_message_by_name(name):
    tmp_path = None
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT filename
                    FROM voice_chat
                    WHERE from_name = %s
                    LIMIT 1;
                    """,
                    (name,)
                )
                row = cur.fetchone()
                if row:
                    tmp_path = row[0]
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
    
    if tmp_path is None:
        return "No message found"
    
    r = sr.Recognizer()

    try:
        # Transcript
        tmp_full = os.path.join(UPLOAD_DIR_CHAT,tmp_path)
        print(tmp_full)
        with sr.AudioFile(tmp_full) as source:
            audio = r.record(source) 

        text = r.recognize_openai(
            audio,
            model="whisper-1"
        )
        return text
    except Exception as e:
        print('[!] Error during transcription:', e)
        return  "Internal Server Error"
def send_voice_radiance(to_name: str, from_uid: int, voice_data: bytes) -> str:
    """
    Send a voice message from `from_uid` to the user named `to_name`.
    Looks up the most recent conversation partner to get their UID and your name.
    """
    try:
        # Acquire a connection + cursor in one go
        with pool.connection() as conn, conn.cursor() as cur:
            # 1) Find the other party's UID (to_uid) and your display name (from_name)
            cur.execute("""
                SELECT
                    CASE
                        WHEN to_name = %s AND from_uid = %s THEN to_uid
                        ELSE from_uid
                    END AS partner_uid,
                    CASE
                        WHEN to_name = %s AND from_uid = %s THEN from_name
                        ELSE to_name
                    END AS your_name
                FROM voice_chat
                WHERE (to_name = %s AND from_uid = %s)
                   OR (from_name = %s AND to_uid = %s)
                ORDER BY created_at DESC
                LIMIT 1
            """, (
                to_name, from_uid,
                to_name, from_uid,
                to_name, from_uid,
                to_name, from_uid
            ))
            row = cur.fetchone()
            if not row:
                print(f"[!] No prior chat found between UID={from_uid} and name='{to_name}'")
                return "User not found"

            to_uid, from_name = row
            print(f"[•] Sending from '{from_name}' (UID={from_uid}) → UID={to_uid} ('{to_name}')")

            # 2) Insert the new voice chat record
            cur.execute("""
                INSERT INTO voice_chat
                    (from_uid, to_uid, from_name, to_name, filename, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                from_uid,
                to_uid,
                from_name,
                to_name,
                voice_data
            ))
            conn.commit()
            print("[+] Voice message sent successfully")
            return "Voice message sent successfully"

    except Exception as e:
        # You might want to log the stack trace in real code
        print(f"[!] Error in send_voice_radiance: {e}")
        return "Internal Server Error"



from elevenlabs.client import ElevenLabs
from elevenlabs import play, VoiceSettings
client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)
import uuid
def compose_message_radi(to_name, from_uid, voice_text):
    response = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb", 
            output_format="mp3_22050_32",
            text=voice_text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
    filename = f"{uuid.uuid4()}.wav"
    print(filename)
    save_file_path = f"uploads_chat/"+filename
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    return send_voice_radiance(to_name, from_uid, filename)


def send_distress_message(from_uid: str,
                          distress_text: str = "This is an emergency. Please help!") -> str:
    """
    Sends a distress voice‐message to the user's emergency contact,
    looking up both the sender’s and contact’s names in the DB.
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # 1) Fetch sender’s name
            cur.execute(
                "SELECT username FROM users WHERE uid = %s",
                (from_uid,)
            )
            sender_row = cur.fetchone()
            if not sender_row:
                return "Sender user not found."
            from_name = sender_row[0]

            # 2) Fetch emergency contact array: [contact_uid, contact_name, contact_phone]
            cur.execute(
                "SELECT emergency_contact FROM user_preferences WHERE user_id = %s",
                (from_uid,)
            )
            pref_row = cur.fetchone()
            if not pref_row or not pref_row[0]:
                return "No emergency contact set for this user."
            contact_uid, contact_name, contact_phone = pref_row[0]

    # 3) Synthesize distress audio
    tts_resp = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_22050_32",
        text=distress_text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    filename = f"{uuid.uuid4()}.mp3"
    save_path = f"uploads_chat/{filename}"
    with open(save_path, "wb") as f:
        for chunk in tts_resp:
            if chunk:
                f.write(chunk)

    # 4) Insert into voice_chat with real from_name
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voice_chat
                  (from_uid, to_uid, from_name, to_name, filename, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (from_uid, contact_uid, from_name, contact_name, filename)
            )
        conn.commit()

    return f"Distress message sent from '{from_name}' to '{contact_name}'."


