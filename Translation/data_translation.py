# Install: pip install requests
import json
import os
import requests

INPUT_FOLDER = r'D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json'
OUTPUT_FOLDER = r'D:\Speech_translation\new_jsons'

# Use public instance or your own: docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
LIBRETRANSLATE_URL = "http://127.0.0.1:5000/translate"  # or https://libretranslate.com/translate

def translate_text(text):
    if not text:
        return ""
    
    payload = {
        "q": text,
        "source": "en",
        "target": "de"
    }
    
    try:
        response = requests.post(LIBRETRANSLATE_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["translatedText"]
    except Exception as e:
        print(f"Error: {e}")
        return ""

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "Files" in data:
        for file_entry in data["Files"]:
            prompt = file_entry.get("Prompt", {})
            transcript = prompt.get("Transcript", "")
            if transcript and "Translation" not in prompt:
                translation = translate_text(transcript)
                prompt["Translation"] = translation
    
    return data

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".json"):
            continue
        
        infile = os.path.join(INPUT_FOLDER, fname)
        try:
            augmented = process_file(infile)
            
            outfile = os.path.join(OUTPUT_FOLDER, fname)
            with open(outfile, "w", encoding="utf-8") as out_f:
                json.dump(augmented, out_f, ensure_ascii=False, indent=2)
            
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {infile}: {e}")

if __name__ == "__main__":
    main()
