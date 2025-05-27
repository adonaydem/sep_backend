from ocr_utils import run_ocr, pipeline_clean_text
import cv2
import os
from PIL import Image
import pytesseract
import time
from langchain_core.prompts import ChatPromptTemplate
def run_ocr(path):
    start_time = time.time()

    img = cv2.imread(path)
    omg = pipeline_clean_text(img)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    langs = 'eng+ara'
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