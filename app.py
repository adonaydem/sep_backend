import os
import tempfile

from flask import Flask, request, jsonify, send_file
import openai
from langchain_openai import ChatOpenAI
import speech_recognition as sr
import os
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import play, VoiceSettings
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)
r = sr.Recognizer()
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # 1. Get the uploaded file
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    audio_file = request.files['audio']
    sid = request.form.get('uid')
    print("TTESING: ", request.form.get('recognitionsCache'))
    if isinstance(request.form.get('recognitionsCache'), str):
        cache = json.loads(request.form.get('recognitionsCache'))
    else:
        cache = request.form.get('recognitionsCache')

    print("!!!!!!!!cache: ",cache, "sid", sid)
    # 2. Save to a temp file so Whisper can read it
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Transcript
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source) 

        text = r.recognize_openai(
            audio,
            model="whisper-1"
        )
        print("____________recognized text: ", text)

        #Response
        text_to_speak = llm_run(text, cache,sid)


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
        
        print(f"{save_file_path}: A new audio file was saved successfully!")
        return send_file(
            save_file_path,
            mimetype="audio/mpeg",
            as_attachment=True,
        )

    except Exception as e:
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import random
def supervisor(inp, cache, sid):
    try:
        cache = {k: v for k, v in cache.items() if v not in [None, ""]} 
        sup_messages = [HumanMessage(content=inp)]
        parent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)
        functions = [convert_to_openai_function(t) for t in toolkit['General']]

        if len(next_app) == 0 or next_app == "No_Match":
            if len(functions) > 0:
                llm = llm.bind_functions(functions)
            # sid = sid + "_" + "No_Match"
            return llm, "", sid
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
    return llm, wanted_prompt, sid 

def llm_run(text, cache, session_id):
    if session_id is None:
        print("!!!!!!!!!session_id is None")
        session_id = random.randint(0, 100000)
    if cache is None:
        print("!!!!!!!!!cache is None")
        cache = {}
    model, prompt, new_sid = supervisor(text, cache, session_id)
    history = MongoDBChatMessageHistory(
        connection_string="mongodb://localhost:27017",  # your MongoDB URI
        session_id=new_sid,                # unique per user/conversation
        database_name="chat_db",                       # optional; defaults to "chat_history"
        collection_name="messages"                     # optional; defaults to "message_store"
    )
    
    messages = history.messages  
    print(messages)
    if len(messages) > 5:
        messages = messages[-5:]
    response_text = model.invoke(messages + [SystemMessage(content=prompt), HumanMessage(content=text)])
    history.add_user_message(text)
    history.add_ai_message(response_text.content)
    print("response: ",response_text.content, type(response_text.content))

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
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
