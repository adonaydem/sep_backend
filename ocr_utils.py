import cv2
import pytesseract

# Load Image
def load_image(path):
    image = cv2.imread(path)
    return image
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened
# Convert to Grayscale
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu Thresholding
def otsu_threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Adaptive Thresholding (Better for uneven lighting)
def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Remove Noise (Non-Local Means Denoising)
def denoise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

# Morphological Operations (Closing Gaps in Characters)
def morph_close(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Resize for Better Recognition
def upscale_image(image, scale=2):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# Edge Detection (Useful for Structured Text, Tables)
def edge_detection(image):
    return cv2.Canny(image, 100, 200)

# Run Tesseract OCR
def run_ocr(image, config="--oem 3 --psm 3"):
    return pytesseract.image_to_string(image, config=config, lang='eng+arb')

def pipeline_clean_text(image):
    gray = to_grayscale(image)
    gray = denoise(gray)
    processed = adaptive_threshold(gray)
    processed = upscale_image(processed)
    return processed