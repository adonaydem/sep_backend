from ocr_utils import run_ocr, pipeline_clean_text
import cv2
import os
from PIL import Image
import pytesseract
import time

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