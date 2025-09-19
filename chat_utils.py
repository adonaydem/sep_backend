from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import FunctionMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph.prebuilt import create_react_agent
import json
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
import os
from langchain_tavily import TavilySearch
from psycopg_pool import ConnectionPool

from langchain_core.tools import tool
from typing import Annotated, List, Optional, Tuple, Union

import ocr
import directions
import scenedescription

from postgres_utils import ResilientPostgresSaver
import voice_chat

load_dotenv()
tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
)
memory=None
try:
    memory = ResilientPostgresSaver()
except Exception as e:
    print(f"[!] Error during postgres: {e}")

llm = ChatOpenAI(
    model="o4-mini"
)


system_prompt = f"""

You are Radiance, a voice-first visual assistant embedded in the Radiance app, an ecosytem built to empower users—especially those with visualimpairments—to experience their surroundings more confidently andindependently.  
Your users are visually impaired, and your goal is to make their daily lives easier by becoming a whole tooling ecosystem for example interpreting images and answering questions in natural, spoken-style language.

Your responses should be readable and no acronyms and shortening settings are allowed.
App Background: 
You have input coming from Application. Note that this is programmatic and nothing to do with user. Maybe the user wants you to use it or maybe not.
Your aim is to look at the user's human text and make judgement about what external data or tool to use. 
Note: If you decide to implement tool calls, do it immediately without permission or notifying user.
You have input data together with Human message: 1. Input JSON for object detection output  2. Image feed captured at the moment of sending message. A flag 'Image sent by user flag' will be provided.

————————————————————————
1. GENERAL BEHAVIOR
————————————————————————
• Always reply as if speaking out loud: use short sentences, contractions, and clear phrasing.  
• Never mention internal data structures (JSON, confidence scores, object classes, etc.).  
• If anything is uncertain, use tentative language (“it looks like,” “it seems,” “as if”).


————————————————————————
2. HANDLING METADATA INPUT
————————————————————————
Only When the user is talking about objects. Eg. “Is there a cat on the table?” "What are these?"
Only available if the system provides image metadata (object-detection JSON tagged with timestamps and confidence scores):

  A.   Object Detection
   1.  Select only the newest data (highest timestamp).  
   2.  Pick up items that are the most recent and reject those that are older than 2 minutes when compared to the most recent.  
   3.  In one sentence, list the items.  
       • Example: “I see a cat, a bicycle, and a coffee cup.”  
   4.  In one follow-up sentence, explain why those objects stand out or matter.  

————————————————————————
3. USING COMPUTER-VISION TOOLS
————————————————————————
When the user explicitly asks for text or scene interpretation, call the appropriate tool:

  A.   Text Recognition (OCR)
   • Use the text_recognition tool to extract raw text.  
   • Clean up merged columns, fix line breaks, and reorganize blocks so the output reads like normal prose.  
   • Read it back in one natural, continuous paragraph.
    Example
    "What is the board saying"
    "Read this out for me."
  B.   Scene Description
   • Use the scene_description tool to get scene category and attributes.  
   • Craft up to two sentences describing the overall scene.  
   • Use simple, everyday words and tentative phrases.  
   • Do not invent any details beyond what the tool returns.

    Example:
    "Can you describe the scene?"
    "Can you describe the environment?"

————————————————————————
4. DETERMINING USER INTENT
————————————————————————
Every time the user submits a message with or without an image:

  1.  Ask yourself: “Is this about interpreting the image?”  
  2.  If yes, decide which mode (object, text, or scene).  
  3.  If it’s unclear which feature they want, ask:  
       “Would you like me to describe objects, read text, or summarize the scene?”  
  4.  If the user’s question is unrelated to image interpretation, switch to normal voice-assistant mode and use tools.
————————————————————————
4. Messenger Feature
————————————————————————
• The software allows users to interact with their peers using voice-controlled messenger.
• You have various tools to handle voice chat and voice-to-text translation.
When the user sends a voice message, you should quote them. Example: You have a message from ___ . They said "Hi, how are you". 

————————————————————————
4. Turn by Turn Navigation Feature
————————————————————————
• The software allows users to recieve turn by turn walking directions using natural languages.
• The aim is to aid visually impaired persons to relieve their daily activities.
• You can use get_walking_directions tool to recieve a list of direction instructions. The tool accepts location name or coordinates(A list containing latitude and longitude) as origin and destination input. 
    In order for the tool to work you need to give detailed information about the location.
Some turn instruction sets can be lengthy so just give them a thorough overview of the path. After which, if they stay interested, you can give turn by turn instructions.
————————————————————————
5. Internet Search
————————————————————————
• The software allows users to search the internet using natural languages. use tavily_tool.
————————————————————————
6. EXAMPLES
————————————————————————

• User: “Can you read this sign?”  
  – Use text_recognition tool → “The sign reads: ‘Welcome to Green Meadows Park.’”

• User: “What do you think this place is?”  
  – Use scene_description tool → “It looks like a busy city street with tall buildings and passing cars.”


    """
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",  # Keep the most recent messages
        token_counter=len,
        # When token_counter=len, each message
        # will be counted as a single token.
        # Remember to adjust for your use case
        max_tokens=20,  # Adjust based on your model's context window
        start_on="human",  # Start trimming from the last human message
        end_on=("human", "tool"),  # End trimming at the last human or tool message
        include_system=True  # Include the initial system message if present
    )

    return {"llm_input_messages": trimmed_messages}

def call_agent(text, input_cache, session_id, tmp_image, latitude=None, longitude=None, address=None,**kwargs):

    @tool
    def text_recognition() -> str:
        "Use this tool to accomplish Manual Text Recognition using OCR, when there is no Data from application."
        try:
            if tmp_image is None:
                print("No image found")
                return "Internal Server Error"
            text = ocr.run_ocr(tmp_image, session_id)
            text = ocr.refine_ocr_util(text, llm)
            return text
        except Exception as e:
            print("OCR Error" + str(e))
            return "Internal Server Error"
    @tool
    def scene_description() -> str:
        "Use this tool to accomplish Manual Scene Description/Categorization, when there is no data from application."
        if kwargs.get('blip_processor') is None:
            print("kwargs None")
            return "Internal Server Error"
        if tmp_image is None:
            print("No image found")
            return "Internal Server Error"
        try:
            text = scenedescription.enhanced_describe(tmp_image,**kwargs)
        except Exception as e:
            text = "Internal Server Error"
            print("Error sd " + str(e))
        return text
    
    @tool
    def get_unread_messages() -> str:
        "Use this tool to get unread messages."
        
        try:
                return voice_chat.get_unread_by_to_uid(session_id)
        except Exception as e:
                print("Failed to get messages", str(e))
                return "Failed to get messages"
         
    @tool
    def get_message_by_id(id : Annotated[str, "message identifier."]) -> str:
        "Use this tool to get message by id."

        try:
            return voice_chat.get_transcripted_message_by_id(id)
        except Exception as e:
            print("Failed ", str(e))
            return "Failed to get message"
    @tool
    def compose_message(to_name : Annotated[str, "Recipient Name."],  message : Annotated[str, "Text Message."]) -> str:
        "Use this tool to send message."
        try:
            return voice_chat.compose_message_radi(to_name, session_id, message)
        except Exception as e:
            print("Failed to send message", str(e))
            return "Failed to send message"
    def get_walking_directions(origin: Annotated[Union[str, List[float]], "Origin as a place name or [longitude, latitude]."],destination: Annotated[str, "Destination as a place name."]) -> str:   
        "Use this tool to get walking directions. Origin and Destination Inputs Required"
        if origin is None or destination is None:
            return "Error:Location Inputs required"
        try:
            return directions.get_walking_instructions(origin, destination)
        except Exception as e:
            print("Failed to get directions", str(e))
            return "Failed to get directions"
    def get_walking_directions_from_current_location(destination: Annotated[str, "Destination as a place name."]) -> str:   
        "Use this tool to get walking directions from user's current location. Destination Input Required"
        if destination is None:
            return "Error:Location Input required"
        if (latitude is None or longitude is None) or address is None:
            return "Error: Couldn't get current location. ask user where they are or to turn on their location.s"
        try:
            if latitude is not None and longitude is not None:
                origin = (longitude,latitude)
            elif address is not None:
                origin = address
            return directions.get_walking_instructions(origin, destination)
        except Exception as e:
            print("Failed to get directions", str(e))
            return "Failed to get directions"
    def get_my_location() -> str:
        "Use this tool to get user's location."
        try:
            return f"Address: {address}" + f"Latitude: {latitude}" + f"Longitude: {longitude}"
        except Exception as e:
            print("Failed to get location", str(e))
            return "Failed to get location"
    @tool
    def send_emergency_message(message : Annotated[Union[str, None], "Text Distress Message"]) -> str:
        "Use this tool to send message."
        try:
            return voice_chat.send_distress_message(session_id, message)
        except Exception as e:
            print("Failed to send message", str(e))
            return "Failed to send message"
    

    
    tools = [tavily_tool, text_recognition, scene_description, get_unread_messages, get_message_by_id,compose_message, get_walking_directions, send_emergency_message, get_my_location, get_walking_directions_from_current_location]
    
    graph = create_react_agent(llm, tools=tools,checkpointer=memory, prompt=system_prompt,pre_model_hook=pre_model_hook,)
    
    graph_backup = create_react_agent(llm, tools=tools, prompt=system_prompt,pre_model_hook=pre_model_hook)
    config = {
        "configurable": {
            "thread_id":  session_id+"15"
        }
    }
    image_found = False if not tmp_image else True
    print("flag", image_found)
    if input_cache is not None:
        if len(input_cache) != 0:
            input_cache = input_cache[::-1]
            input_cache = input_cache[:3]
    try:
        response = graph.invoke({"messages": [SystemMessage(f"Input JSON data: \n{json.dumps(input_cache, indent=4)}" + f"\nImage sent by user flag: {image_found} "), HumanMessage(text)]}, config)
    except Exception as e:
        print("GRAPH e: ",e)
        response = graph_backup.invoke({"messages": [SystemMessage(f"Input JSON data: \n{json.dumps(input_cache, indent=4)}"+ f"\nImage sent by user flag: {image_found} "), HumanMessage(text)]}, config)
    finally:
        memory.close()
    print(response['messages'])
    return response['messages'][-1].content



