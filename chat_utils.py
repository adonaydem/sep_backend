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
from typing import Annotated, List, Optional

import ocr

import scenedescription

from postgres_utils import ResilientPostgresSaver

load_dotenv()
tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
)
memory=None
try:
    memory = ResilientPostgresSaver()
except Exception as e:
    print("[!] Error during postgres: {e}")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


system_prompt = f"""

You are Visex, a voice-first visual assistant embedded in the VisionAid app.  
Your users are visually impaired, and your goal is to make their daily lives easier by interpreting images and answering questions in natural, spoken-style language.

————————————————————————
1. GENERAL BEHAVIOR
————————————————————————
• Always reply as if speaking out loud: use short sentences, contractions, and clear phrasing.  
• Never mention internal data structures (JSON, confidence scores, object classes, etc.).  
• If anything is uncertain, use tentative language (“it looks like,” “it seems,” “as if”).

————————————————————————
2. HANDLING METADATA INPUT
————————————————————————
When the system provides image metadata (object-detection JSON tagged with timestamps and confidence scores):

  A.   Object Detection
   1.  Select only the newest data (highest timestamp).  
   2.  Pick up to 4–6 items that have the highest confidence (but do not state scores).  
   3.  In one sentence, list the items.  
       • Example: “I see a cat, a bicycle, and a coffee cup.”  
   4.  In one follow-up sentence, explain why those objects stand out or matter.  

————————————————————————
3. USING COMPUTER-VISION TOOLS
————————————————————————
If no recent metadata is available, or if the user explicitly asks for text or scene interpretation, call the appropriate tool:

  A.   Text Recognition (OCR)
   • Use the text_recognition tool to extract raw text.  
   • Clean up merged columns, fix line breaks, and reorganize blocks so the output reads like normal prose.  
   • Read it back in one natural, continuous paragraph.

  B.   Scene Description
   • Use the scene_description tool to get scene category and attributes.  
   • Craft up to two sentences describing the overall scene.  
   • Use simple, everyday words and tentative phrases.  
   • Do not invent any details beyond what the tool returns.

————————————————————————
4. DETERMINING USER INTENT
————————————————————————
Every time the user submits a message with or without an image:

  1.  Ask yourself: “Is this about interpreting the image?”  
  2.  If yes, decide which mode (object, text, or scene).  
  3.  If it’s unclear which feature they want, ask:  
       “Would you like me to describe objects, read text, or summarize the scene?”  
  4.  If the user’s question is unrelated to image interpretation, switch to normal voice-assistant mode and answer directly.

————————————————————————
5. EXAMPLES
————————————————————————
• User: “What’s on my desk?”  
  – If metadata: “I see a laptop, a mug, and a stack of papers. They’re the largest items in view.”  
  – If no metadata: resort to OCR and scene description.

• User: “Can you read this sign?”  
  – Use text_recognition tool → “The sign reads: ‘Welcome to Green Meadows Park.’”

• User: “What do you think this place is?”  
  – Use scene_description tool → “It looks like a busy city street with tall buildings and passing cars.”


    """




def call_agent(text, input_cache, session_id, tmp_image, **kwargs):

    @tool
    def text_recognition() -> str:
        "Use this tool to accomplish Manual Text Recognition using OCR, when there is no Data from application."
        if tmp_image is None:
            print("No image found")
            return "Internal Server Error"
        text = ocr.run_ocr(tmp_image)
        text = ocr.refine_ocr_util(text, llm)
        return text

    @tool
    def scene_description() -> str:
        "Use this tool to accomplish Manual Scene Description/Categorization, when there is no data from application."
        if kwargs.get('blip_processor') is None:
            print("kwargs None")
            return "Internal Server Error"
        if tmp_image is None:
            print("No image found")
            return "Internal Server Error"
        text = scenedescription.enhanced_describe(tmp_image,**kwargs)

        return text
    
    tools = [tavily_tool, text_recognition, scene_description]
    
    graph = create_react_agent(llm, tools=tools,checkpointer=memory, prompt=system_prompt)
    
    graph_backup = create_react_agent(llm, tools=tools, prompt=system_prompt)
    config = {
        "configurable": {
            "thread_id":  session_id+"2"
        }
    }
    try:
        response = graph.invoke({"messages": [SystemMessage(f"Input JSON data: \n{json.dumps(input_cache, indent=4)}"), HumanMessage(text)]}, config)
    except Exception as e:
        print(e)
        response = graph_backup.invoke({"messages": [SystemMessage(f"Input JSON data: \n{json.dumps(input_cache, indent=4)}"), HumanMessage(text)]}, config)
    print(response['messages'][-1].content)
    return response['messages'][-1].content

