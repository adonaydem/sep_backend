import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import models, transforms
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load Models with enhanced error handling
def load_models():
    print("Loading BLIP...")
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        raise RuntimeError(f"Failed to load BLIP models: {e}")

    print("Loading Places365...")
    try:
        places_model = models.resnet18(num_classes=365)
        model_path = os.path.join(SCRIPT_DIR, "resnet18_places365.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)  # Handle different save formats
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
        places_model.load_state_dict(state_dict)
        places_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load Places365 model: {e}")

    # Load categories with robust parsing
    categories_path = os.path.join(SCRIPT_DIR, "categories_places365.txt")
    try:
        with open(categories_path) as f:
            places_classes = []
            for line in f.readlines():
                parts = line.strip().split(' ')
                if len(parts) >= 1:
                    category = parts[0][3:] if parts[0].startswith('/') else parts[0]
                    places_classes.append(category)
            
            if not places_classes:
                raise ValueError("No valid categories found in categories file")
            print(f"Successfully loaded {len(places_classes)} categories")
    except Exception as e:
        raise RuntimeError(f"Failed to load categories: {e}")

    return blip_processor, blip_model, places_model, places_classes

# 2. Enhanced prediction function with error handling
def get_places_prediction(image, model, classes):
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
        
        _, pred = torch.max(outputs, 1)
        pred_idx = pred.item()
        if pred_idx >= len(classes):
            print(f"Warning: Predicted index {pred_idx} out of range (max {len(classes)-1})")
            return "unknown location"
        return classes[pred_idx]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown location"

# 3. Main description function with comprehensive error handling
def enhanced_describe(image_path, blip_processor, blip_model, places_model, places_classes):
    try:
        # Convert to absolute path
        abs_image_path = os.path.join(SCRIPT_DIR, image_path)
        print(f"Processing image at: {abs_image_path}")
        
        # Verify image exists
        if not os.path.exists(abs_image_path):
            available_files = [f for f in os.listdir(SCRIPT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            raise FileNotFoundError(
                f"Image '{image_path}' not found in {SCRIPT_DIR}\n"
                f"Available images: {available_files or 'None found'}"
            )

        # Open and verify image
        try:
            image = Image.open(abs_image_path).convert('RGB')
            image.verify()  # Verify image integrity
            image = Image.open(abs_image_path).convert('RGB')  # Reopen after verify
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        # BLIP description
        try:
            inputs = blip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                output = blip_model.generate(**inputs)
            
            if len(output) == 0:
                blip_desc = "No description generated"
            else:
                blip_desc = blip_processor.decode(output[0], skip_special_tokens=True) or "No description"
        except Exception as e:
            print(f"BLIP error: {e}")
            blip_desc = "Could not generate description"

        # Places365 prediction
        place_context = get_places_prediction(image, places_model, places_classes)

        # Combine results
        if place_context.lower() == "unknown location":
            return blip_desc, ""
        if place_context.lower() in blip_desc.lower():
            return blip_desc, ""
        return place_context, blip_desc

    except Exception as e:
        print(f"Error in scene description: {e}")
        return "Could not generate description"

# 4. Main execution with proper error handling
if __name__ == "__main__":
    try:
        print("Initializing models...")
        blip_processor, blip_model, places_model, places_classes = load_models()
        
        # Test with available images
        test_images = [f for f in os.listdir(SCRIPT_DIR) if f.lower().endswith(('.jpeg', '.jpg'))]
        
        if not test_images:
            print("No test images found in directory. Please add a JPG/PNG image.")
        else:
            print(f"\nFound test images: {test_images}")
            for img_file in test_images:
                print(f"\nProcessing {img_file}...")
                description = enhanced_describe(img_file, blip_processor, blip_model, places_model, places_classes)
                print("\nSCENE DESCRIPTION:", description)
                
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Verify all required files are present:")
        print("   - resnet18_places365.pth")
        print("   - categories_places365.txt")
        print("   - At least one test image (JPG/PNG)")
        print(f"2. Current directory contents: {os.listdir(SCRIPT_DIR)}")
        print("3. Check file permissions")