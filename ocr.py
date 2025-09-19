from ocr_utils import run_ocr, pipeline_clean_text
import cv2
import os
from PIL import Image
import pytesseract
import time
from langchain_core.prompts import ChatPromptTemplate
import psycopg as pg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv('POSTGRES_SUPABASE')  # e.g. 'postgresql://user:pass@host:port/db'


def fetch_langs(user_id):
    seq = ""
    try:
        with pg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT languages
                    FROM user_preferences
                    WHERE user_id = %s;""",
                    (user_id,)
                )
                row = cur.fetchone()
                print(row[0])
                if not row:
                    return "eng+ara"
        for lang in row[0]:
            if lang.strip() == "":
                continue
            if lang.strip().lower() == "german":
                seq += "+deu"
            else:
                seq += "+" + lang.strip().lower()[:3]
        print(seq)
        return seq[1:]
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
        return "eng+ara"
def run_ocr(path,uid):
    start_time = time.time()

    img = cv2.imread(path)
    img = pipeline_clean_text(img)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    langs = fetch_langs(uid)
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
    text = pytesseract.image_to_string(img, lang=langs)
    print(text, time.time() - start_time)
    return text


def refine_ocr_util(text,llm):
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
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-4.1-mini')
# print(refine_ocr_util(run_ocr('swe.jpg', "zTwzboqdazPtxW14qMdgxmZoNj83"), llm))