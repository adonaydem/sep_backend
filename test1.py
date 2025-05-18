from langchain_openai import ChatOpenAI

import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
messages = [
    
]
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
            print("res",response_text)
            engine.say(response_text.content)
            engine.runAndWait()
        

        

listen()