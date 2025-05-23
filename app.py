import os
import tempfile

from flask import Flask, request, jsonify, send_file
import openai
from langchain_openai import ChatOpenAI, OpenAI
import speech_recognition as sr
import os
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import play, VoiceSettings
from dotenv import load_dotenv

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import random
from langchain_postgres import PostgresChatMessageHistory
import psycopg
import uuid
from langchain.chains.transform import TransformChain
from langchain.prompts import PromptTemplate
from ocr import run_ocr
import logging

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import models, transforms
import os
import scenedescription
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

sync_connection = psycopg.connect(
    os.getenv("POSTGRES_SUPABASE"),
    prepare_threshold=None,
)
name_space = uuid.NAMESPACE_URL
client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)
r = sr.Recognizer()

blip_processor, blip_model, places_model, places_classes = scenedescription.load_models()
logger.info("SD models loaded")

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # 1. Get the uploaded file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    audio_file = request.files['audio']
    sid = request.form.get('uid')
    logger.debug("TTESING: %s", request.form.get('recognitionsCache'))
    if isinstance(request.form.get('recognitionsCache'), str):
        cache = json.loads(request.form.get('recognitionsCache'))
    else:
        cache = request.form.get('recognitionsCache')

    logger.debug("!!!!!!!!cache: %s sid %s", cache, sid)
    # 2. Save to a temp file so Whisper can read it
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    logger.debug("Audio file saved to: %s", tmp_path)
    try:
        # Transcript
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source) 

        text = r.recognize_openai(
            audio,
            model="whisper-1"
        )
        logger.debug("____________recognized text: %s", text)

        #Response
        text_to_speak = llm_run(text, cache, sid)

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
        logger.debug("Error: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def supervisor(inp, cache):
    try:
        cache = {k: v for k, v in cache.items() if v not in [None, ""]} 
        sup_messages = [HumanMessage(content=inp)]
        parent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
        toolkit = {member: [] for member in ["Computer_Vision", "General"]}
        members = ["Computer_Vision", "No_Match"]
        cache_key_json = json.dumps([k for k,_ in cache.items()]).replace("{", "{{").replace("}", "}}")
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers:  {json.dumps(members)}. Given the following user request and below addtional user input data, considering the whole conversation history,"
            " respond with the worker that can most probably handle it. If the user request doesn't match with the available wrokers at all, respond with No_Match. Each worker will perform a"
            " task and respond with their results and status. "
            f"Additional User Input data that is available: {cache_key_json[:100]}"
        )
        print("Sys: ",system_prompt)
        options = members
        # Using openai function calling can make output parsing easier for us
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    "Select one of: {options}",
                ),
            ]
        ).partial(options=str(options), members=", ".join(members))


        supervisor_chain = (
            prompt
            | parent_llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
        print("_________supervisor_chain: ", sup_messages)
        response = supervisor_chain.invoke({"messages": sup_messages})
        sup_messages.append(AIMessage(content=str(response)))
        next_app = response['next']
        print("_________?????????????next app: ", next_app)
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5, streaming=True)
        functions = [convert_to_openai_function(t) for t in toolkit['General']]

        if len(next_app) == 0 or next_app == "No_Match":
            if len(functions) > 0:
                llm = llm.bind_functions(functions)
            # sid = sid + "_" + "No_Match"
            return llm, ""
        print("Cache", cache)
        # Bind tools to the model
        functions.extend([convert_to_openai_function(t) for t in toolkit[next_app]])
        if next_app == "Computer_Vision":
            wanted_prompt = f"""
            You are Visio – an expert visual assistant with human-level understanding of images. You will recieve metadata and You will act as if you “see” what’s in front of you and describe it naturally, vividly, and concisely, as if you were speaking to someone who can’t see. You MUST never mention technical metadata (e.g. “JSON,” “classes,” “confidence scores,” etc.). Instead, you speak in plain, engaging language.

            Among the below given input JSON data, exactly one of these three modes applies:
            • OCR (text recognition)
            • Scene_Description
            • Object_Detection

            Your responses must follow these rules:

            1. OCR  
             Read aloud the text as it appears, preserving original punctuation and line breaks.  
             If any snippet is garbled or unclear, offer its best reconstruction or note, “unreadable here.”  
             Output as a clear paragraph.

            2. Scene_Description  
             Turn the raw scene into a vivid, sensory narrative.  
             Mention colors, shapes, spatial arrangement, lighting, mood, and any notable details, If available.  
             Write one flowing descriptive paragraph.

            3. Object_Detection  
             Imagine you’re seeing through a camera: list only the most recent (i.e. based on timestamp) objects and, among those, emphasize on the ones with high confidence score(don't talk about scores explicitly though).  
             Keep it concise—no more than 4–6 items, total of 2-3 sentences.
             No scene related information or tone should be given, as this is about individual items and their metadata.

            If the input doesn’t match any of those modes, reply:  
            “I’m sorry, I can’t interpret what I see at the moment.”

            Input JSON data: 
            {json.dumps(cache, indent=4)}
            """
        else:
            wanted_prompt= ""

        print("!!!!!!!!!prompt", wanted_prompt)
        if len(functions) > 0:
            llm = llm.bind_functions(functions)
        # sid = sid+"_"+next_app
    except Exception as e:
        print("Error: ", str(e))
        raise e
    return llm, wanted_prompt

def llm_run(text, cache, session_id):
    if session_id is None:
        print("!!!!!!!!!session_id is None")
        session_id = random.randint(0, 100000)
    if cache is None:
        print("!!!!!!!!!cache is None")
        cache = []
    # llm, wanted_prompt  = supervisor(text,cache)
    print("!!!!!!!!!cache: ", cache)
    if len(cache) == 0:
        cache = []
    else: 
        cache = cache[-7:]
    
   
    prompt = f"""
        You are Visex – an expert visual assistant with human-level understanding of images. You will recieve metadata and You will act as if you “see” what’s in front of you and describe it naturally, vividly, and concisely, as if you were speaking to someone who can’t see. You MUST never mention technical metadata (e.g. “JSON,” “classes,” “confidence scores,” etc.). Instead, you speak in plain, engaging language.
        Task 1: Object_Detection  
             Imagine you’re seeing through a camera: list only the most recent (i.e. based on timestamp) objects and, among those, emphasize on the ones with high confidence score(don't talk about scores explicitly though).  
             Keep it concise—no more than 4–6 items. Use two sentences only. You should list items in first sentence, then describe consicely the inituition behind these objects.
             No scene related information or tone should be given, as this is about individual items and their metadata.
             If no data is given say “I’m sorry, I can’t interpret what I see at the moment.”
        Input JSON data: 
            {json.dumps(cache, indent=4)}
    """
    

    sync_connection.autocommit = True
    logger.debug("TTESING sid: %s (%s)", session_id, type(session_id))
    session_uuid = str(uuid.uuid5(name_space, str(session_id)))
    history = PostgresChatMessageHistory(  # your MongoDB URI
        "home_chat_history",
        session_uuid,     
        sync_connection=sync_connection
    )
    try:

        messages = history.messages  
        print(messages)
        if len(messages) > 5:
            messages = messages[-5:]
        response_text = llm.invoke(messages + [SystemMessage(content=prompt), HumanMessage(content=text)])
        history.add_messages(
            [
            HumanMessage(content=text),
            response_text
            ]
        )
        print("response: ",response_text.content, type(response_text.content))
    except Exception as e:
        sync_connection.rollback()     # abandon the failed transaction
        raise e
    return response_text.content


@app.route('/chat', methods=['POST'])
def chat_test():
    request_data = request.get_json()
    text = request_data['text']
    sid = request_data['session_id']
    if isinstance(request_data['cache'], str):
        cache = json.loads(request_data['cache'])
    else:
        cache = request_data['cache']
    response_text = llm_run(text, cache, sid)
    return jsonify(response_text)

@app.route('/refine_ocr', methods=['POST'])
def refine_ocr():
    text = request.form.get('text')
    if text.strip() == "":
        ""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert OCR Output corrector. You are proficient in both Arabic and English languages. Possible input languages are English, Arabic or Both at the same time. You are required to correct the output to the point that the block makes sense. 
                   In addition, the current OCR has no spatial awareness so you are required to correct the output to the point that the block makes sense. For example, Two columns in an image will get attached and be seen as one. Therefore, you need to analyze the whole text structure and try to grasp its message.
                    Input: raw OCR output containing Latin, Arabic, and possible garbled symbols.
                    Output: clean Text Output only—no notes or commentary.
                    
                    Example 1:
                    Input: "thttps://examplé.com\nنص م***:*خرب"
                    Output: "https://examplé.com\nنص مخرب"

                    Example 2:
                    Input: "he followingg link: !http://test.org\nالنص 123## مترجم"
                    Output: "The following link: http://test.org\nالنص مترجم"
                    
                    """,
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm
    print("Text Refine input: ", text)
    response_text = chain.invoke({"text": text})
    print("REFINE OUTPUT: ", response_text)
    return response_text.content

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file sent'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    text = run_ocr(filepath)
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

        scene_catagory, scene_attribute = scenedescription.enhanced_describe(filepath, blip_processor, blip_model, places_model, places_classes)
        

        if scene_catagory.strip() == "" and scene_attribute == "":
            return ""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert Image Caption Writer for a visually impaired person. Given a minimal input—scene category and scene attribute—write a natural, descriptive caption as if you are looking directly at the image. Use everyday, simple language to bring the scene to life. Avoid adding details not provided. Write no more than three sentences. Use tentative language such as “it looks like,” “it seems,” “it is as if,” or similar phrases to express some uncertainty or doubt about what you see.
                        Input: raw scene description with minimal details.
                        Output: clean Text Description Output only—no notes or commentary.
                        
                        Example 1:
                        Scene Category: "athletic_field/outdoor."
                        Scene Attribute: " a basketball court"
                        Output: "It seems you are standing near an outdoor basketball court on an athletic field. The court looks like it has clear lines marking the playing area, with a hoop standing at one end. It is as if the space is ready for a game on a bright day."

                        Example 2:
                        Scene Category: "car_interior"
                        Scene Attribute: "  the interior of the 2019 bmw e - tr"
                        Output: "It looks like you are inside the cabin of a 2019 BMW iE electric car. The dashboard seems sleek and modern, with a large digital display in front of the driver’s seat. The seats and controls give an impression of comfort and high-tech design."
                        
                        """,
                ),
                ("human", """

                Scene Category: "{scene_category}"
                Scene Attribute: "{scene_attribute}"
                """),
            ]
        )

        chain = prompt | llm
        print("Text Refine input: ", scene_attribute+scene_catagory)
        response_text = chain.invoke({"scene_category": scene_catagory, "scene_attribute": scene_attribute})
        print("REFINE OUTPUT: ", response_text)
        return jsonify({"text":response_text.content})
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
