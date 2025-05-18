from langchain_openai import ChatOpenAI
from langchain_community.tools import ElevenLabsText2SpeechTool
import speech_recognition as sr
import pyttsx3
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play
os.environ["ELEVENLABS_API_KEY"] = "sk_288315f54a23d59d795d0baccef49ed251a315b252b03a70"
engine = pyttsx3.init()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
messages = []
tts = ElevenLabsText2SpeechTool()


client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5

        print("Okay, go!")
        while 1:
            text = ""
            print("listening now...")
        
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
            print("Recognizing...")
            # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
            # other speech recognition models are also available.
            text = r.recognize_openai(
                audio,
                model="whisper-1"
            )
            print("text", text)
            input()
            response_text = llm.invoke(messages + [{"role": "user", "content": text}])
            print("res",response_text.content, type(response_text.content))
            text_to_speak=response_text.content[:30]
            
            audio = client.text_to_speech.convert(
                text=text_to_speak,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            with open("output.mp3", "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            # speech_file = tts.run(text_to_speak)
            # print(speech_file)
            # tts.play(speech_file)

            # engine.say(response_text.content)
            # engine.runAndWait()
        

        

listen()