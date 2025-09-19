import os
import tempfile

from flask import Flask, request, jsonify, send_file,url_for, send_from_directory
import openai
from langchain_openai import ChatOpenAI, OpenAI
import speech_recognition as sr
import os
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import play, VoiceSettings
from dotenv import load_dotenv

import json
import random
from langchain_postgres import PostgresChatMessageHistory
import psycopg
import uuid
from langchain.chains.transform import TransformChain
from langchain.prompts import PromptTemplate
from ocr import *
import logging

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import models, transforms
import os
import scenedescription
import chat_utils

from werkzeug.utils import secure_filename
UPLOAD_DIR_CHAT = os.path.join(os.path.dirname(__file__), 'uploads_chat')
os.makedirs(UPLOAD_DIR_CHAT, exist_ok=True)
import voice_chat

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="o4-mini",
)

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)
r = sr.Recognizer()
from psycopg_pool import ConnectionPool
pool = ConnectionPool(conninfo=os.getenv('POSTGRES_SUPABASE')) 
blip_processor, blip_model, places_model, places_classes = scenedescription.load_models()
logger.info("SD models loaded")

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/chat_audio', methods=['POST'])
def chat_audio():
    # 1. Get the uploaded file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    audio_file = request.files['audio']
    image_file = request.files['image']
    sid = request.form.get('uid')
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    address = request.form.get('address')
    print("SID ", sid)
    logger.debug("TTESING: ID  %s",request.form.get('recognitionsCache'))
    if isinstance(request.form.get('recognitionsCache'), str):
        cache = json.loads(request.form.get('recognitionsCache'))
    else:
        cache = request.form.get('recognitionsCache')

    logger.info("!!!!!!!!cache: %s sid %s", cache, sid)
    # 2. Save to a temp file so Whisper can read it
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    logger.info("Audio file saved to: %s", tmp_path)
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image_file.filename)[1], delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path_image = tmp.name
    logger.info("image file saved to: %s", tmp_path_image)
    try:
        # Transcript
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source) 

        text = r.recognize_openai(
            audio,
            model="whisper-1"
        )
        logger.info("____________recognized text: %s", text)
        if text is None:
            return jsonify({"error": "No text recognized."}), 400
        #Response
        text_to_speak = chat_utils.call_agent(text, cache, sid, tmp_path_image, latitude=latitude, longitude=longitude, address=address, blip_processor=blip_processor, blip_model=blip_model, places_model=places_model, places_classes=places_classes, llm=llm)

        #TTS
        response = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb", 
            output_format="mp3_22050_32",
            text=text_to_speak,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        save_file_path = f"audio_cache/{uuid.uuid4()}.mp3"
        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
        
        logger.debug("%s: A new audio file was saved successfully!", save_file_path)
        return send_file(
            save_file_path,
            mimetype="audio/mpeg",
            as_attachment=True,
        )

    except Exception as e:
        logger.info("Error: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
@app.route('/tts', methods=['POST'])
def transcribe():
    text_to_speak = request.form.get('text')
    response = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb", 
            output_format="mp3_22050_32",
            text=text_to_speak,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
    save_file_path = f"audio_cache/{uuid.uuid4()}.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    
    logger.debug("%s: A new audio file was saved successfully!", save_file_path)
    return send_file(
        save_file_path,
        mimetype="audio/mpeg",
        as_attachment=True,
    )

@app.route('/chat', methods=['POST'])
def chat_test():
    
    image_file = request.files['image']
    sid = request.form.get('uid')
    cache = request.form.get('cache')
    text = request.form.get('text')
    if isinstance(cache, str):
        cache = json.loads(cache)
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image_file.filename)[1], delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path_image = tmp.name
    text_to_speak = chat_utils.call_agent(text, cache, sid, tmp_path_image, blip_processor=blip_processor, blip_model=blip_model, places_model=places_model, places_classes=places_classes, llm=llm)

    return jsonify(text_to_speak)

@app.route('/refine_ocr', methods=['POST'])
def refine_ocr():
    text = request.form.get('text')
    
    return refine_ocr_util(text, llm)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file sent'}), 400

    file = request.files['image']
    uid = request.form.get('uid')
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    text = run_ocr(filepath,uid)
    print("SENDING TO flutter:", text)
    return jsonify({'extracted_text': text})

@app.route('/sd', methods=['POST'])
def sd():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file sent'}), 400

        file = request.files['image']
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        response_text = scenedescription.enhanced_describe(filepath, blip_processor, blip_model, places_model, places_classes, llm)
        

        
        return jsonify({"text":response_text})
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/send_voice_chat', methods=['POST'])
def send_voice_chat():
    try:
        from_uid = request.form.get('from_uid')
        from_name = request.form.get('from_name')
        to_name = request.form.get('to_name')
        to_uid = request.form.get('to_uid')
        f = request.files.get('voice')

        # Check if required data is present
        if not from_uid or not to_uid or not f:
            return jsonify({'error': 'Missing required fields'}), 400

        # Prefix with a UUID to avoid collisions
        filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        save_path = os.path.join(UPLOAD_DIR_CHAT, filename)
        f.save(save_path)

        # INSERT metadata via supabase-py
        res = voice_chat.insert_file(from_name,to_name,from_uid, to_uid, filename)
        if 'Error' in res:
            return jsonify({'error': res['Error']}), 500

        return ('', 204)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/send_file_chat', methods=['POST'])
def send_file_chat():
    try:
        from_uid = request.form.get('from_uid')
        from_name = request.form.get('from_name')
        to_name = request.form.get('to_name')
        to_uid = request.form.get('to_uid')
        f = request.files.get('file')

        # Check if required data is present
        if not from_uid or not to_uid or not f:
            return jsonify({'error': 'Missing required fields'}), 400

        # Prefix with a UUID to avoid collisions
        filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        save_path = os.path.join(UPLOAD_DIR_CHAT, filename)
        f.save(save_path)

        # INSERT metadata via supabase-py
        res = voice_chat.insert_file(from_name,to_name,from_uid, to_uid, filename, ftype="file")
        if 'Error' in res:
            return jsonify({'error': res['Error']}), 500

        return ('', 204)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/poll_unread', methods=['GET'])

def poll_messages():
    
    

    
    to_uid = request.form.get('to_uid')
    items = voice_chat.get_unread_by_to_uid(to_uid)

    # mark them delivered
    

    # attach URLs
    for it in items:
        it['url'] = url_for('media', filename=it['filename'], _external=True)

    return jsonify(items)

@app.route('/poll_from_uid', methods=['GET'])
def poll_from_uid():
    
    

    from_uid = request.args.get('from_uid')
    to_uid = request.args.get('to_uid')
    items = voice_chat.get_chat_history(to_uid, from_uid)

    # mark them delivered
    

    # attach URLs
    for it in items:
        it['url'] = url_for('media', filename=it['filename'], _external=True)

    return jsonify(items)

@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(UPLOAD_DIR_CHAT, filename)

@app.route('/mark_read', methods=['PUT'])
def read():
    id = request.form.get('id')
    voice_chat.mark_read(id)
    return ('', 204)

@app.route('/list_chat', methods=['GET'])
def list_chat():
    try:
        to_uid = request.args.get('to_uid')

        items = voice_chat.get_chat_list(to_uid)
        print("list chat", items, "to_uid", to_uid)
        return jsonify(items)
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
        return jsonify([])
    


@app.route('/search_messages', methods=['GET'])
def search_messages():
    # grab the text query
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Missing query parameter `q`'}), 400

    # perform the search (implement this in your voice_chat module)
    # should return a list of dicts with at least: id, from_uid, filename, created_at
    results = voice_chat.search_messages(query)

    # attach URLs for each result
    for msg in results:
        msg['url'] = url_for('media', filename=msg['filename'], _external=True)

    return jsonify(results), 200
from preferences import get_preferences, save_preferences

def json_error(message, code=400):
    return jsonify({'error': message}), code

@app.route('/api/preferences', methods=['POST'])
def api_save_preferences():
    data = request.get_json() or {}
    try:
        save_preferences(
            user_id=data['user_id'],
            languages=data.get('languages', []),
            objects=data.get('objects', []),
            voice_speed=data.get('voice_speed', 1.0)
        )
        return jsonify({'status': 'ok'})
    except KeyError:
        return json_error('Missing required field'), 422
    except Exception as e:
        return json_error(str(e), 500)

@app.route('/api/preferences', methods=['GET'])
def api_get_preferences():
    user_id = request.args.get('user_id')
    if not user_id:
        return json_error('user_id query param required', 422)
    try:
        prefs = get_preferences(user_id)
        if prefs is None:
            return jsonify({}), 204
        return jsonify(prefs)
    except Exception as e:
        return json_error(str(e), 500)
 

DB_URL = os.getenv('POSTGRES_SUPABASE')

def get_user_id(phone: str, username: str):
    """Fetch a user's ID based on phone and username."""
    query = 'SELECT uid FROM users WHERE phone_number = %s AND username = %s'
    with pg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (phone, username))
            row = cur.fetchone()
    print("ROW",row)
    return row[0] if row else None


def get_emergency_contact_db(user_id):
    """Fetch emergency_contact JSON for a given user_id."""
    print(user_id, type(user_id))
    try:
        with pg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT emergency_contact FROM user_preferences WHERE user_id = %s',
                    (user_id,)
                )
                row = cur.fetchone()
        print(row)
        if not row or row[0] is None:
            return None
        return row[0]
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
        return None

def upsert_emergency_contact(user_id, contact_uid, contact_name, user_phone) -> None:
    """Insert or update the emergency_contact for a user."""
    # Check if user exists
    
    record = [contact_uid, contact_name, user_phone]
    user_exists_sql = "SELECT 1 FROM user_preferences WHERE user_id = %s"
    with pg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(user_exists_sql, (user_id,))
            if cur.fetchone() is None:
                upsert_sql = """
                    INSERT INTO user_preferences (user_id, emergency_contact)
                    VALUES (%s, %s::text[])
                    ON CONFLICT (user_id) DO UPDATE SET
                    emergency_contact = EXCLUDED.emergency_contact,
                    updated_at = NOW();
                """
                with pg.connect(DB_URL) as conn:
                    with conn.cursor() as cur:
                        cur.execute(upsert_sql, (user_id, record))
                    conn.commit()
            else:
                upsert_sql = """
                    UPDATE user_preferences
                    SET emergency_contact = %s::text[]
                    WHERE user_id = %s;
                """
                with pg.connect(DB_URL) as conn:
                    with conn.cursor() as cur:
                        cur.execute(upsert_sql, (record, user_id))
                    conn.commit()

    
    


@app.route('/emergency_info', methods=['GET'])
def get_emergency_contact():
    user_id = request.args.get('uid')
    if not user_id:
        print("!!!!!Error: User ID is null")
        return jsonify({'error': 'User ID is required'}), 400

    contact = get_emergency_contact_db(user_id)
    if contact is not None:
        contact = {"phone": contact[2], "name": contact[1]}
        return jsonify(contact), 200

    return jsonify({'error': 'No contact found'}), 404
@app.route('/emergency_info', methods=['POST'])
def set_emergency_contact():
    # 1) Validate inputs
    my_uid_raw = request.form.get('uid')
    print("uid", my_uid_raw)
    if not my_uid_raw:
        return jsonify({'error': 'Your user-id (uid) is required'}), 400
    

    contact_phone = request.form.get('phone')
    contact_name  = request.form.get('name')
    print("inp",contact_phone, contact_name)
    if not contact_phone or not contact_name:
        return jsonify({'error': 'Both name and phone are required'}), 400

    # 2) Lookup contact user's internal ID
    contact_uid = get_user_id(contact_phone, contact_name)
    if contact_uid is None:
        return jsonify({'error': 'Contact user not found'}), 404

    # 3) Finally upsert
    upsert_emergency_contact(my_uid_raw, str(contact_uid), contact_name, contact_phone)
    return jsonify({'status': 'saved'}), 201


@app.route('/distress', methods=['POST'])
def distress_endpoint():
    user_uid = request.form.get('uid')
    if not user_uid:
        return jsonify({'error': 'uid is required'}), 400

    # you could accept custom text too:
    text = request.form.get('text') or "I need help right now!"
    result = voice_chat.send_distress_message(user_uid, text)
    status = 200 if "sent" in result.lower() else 400
    return jsonify({'status': result}), status

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    uid = data.get('uid')
    email = data.get('email')
    username = data.get('username')

    if not uid or not email or not username:
        return jsonify({'error': 'Missing uid or email'}), 400

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (uid, username,email) VALUES (%s,%s, %s) ON CONFLICT (uid) DO NOTHING",
                    (str(uid), str(username),str(email))
                )
        return jsonify({'message': 'User registered successfully'}), 200
    except Exception as e:
        print(f"[!] Error during INSERT: {e}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # bind to 0.0.0.0 so Fly’s proxy can reach you
    app.run(host="0.0.0.0", port=port, debug=True)

