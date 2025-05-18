from langchain_openai import ChatOpenAI
import speech_recognition as sr
import os
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs import play, VoiceSettings
os.environ["ELEVENLABS_API_KEY"] = "sk_288315f54a23d59d795d0baccef49ed251a315b252b03a70"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
messages = []

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

r = sr.Recognizer()    
with sr.AudioFile("./000fc969.wav") as source:
    audio = r.record(source) 

text = r.recognize_openai(
    audio,
    model="whisper-1"
)
print("text", text)

response_text = llm.invoke(messages + [{"role": "user", "content": text}])
print("res",response_text.content, type(response_text.content))
text_to_speak=response_text.content[:100]

response = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text_to_speak,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        # Optional voice settings that allow you to customize the output
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
)
# uncomment the line below to play the audio back
# play(response)
# Generating a unique file name for the output MP3 file
save_file_path = f"{uuid.uuid4()}.mp3"
# Writing the audio to a file
with open(save_file_path, "wb") as f:
    for chunk in response:
        if chunk:
            f.write(chunk)
print(f"{save_file_path}: A new audio file was saved successfully!")
