# Install: pip install pygoogletranslation
import json
import os
import asyncio
from googletrans import Translator


INPUT_FOLDER = r'D:\Speech_translation\SpeechAccessibility_2025-11-02_Train_Only_Json'
OUTPUT_FOLDER = r'D:\Speech_translation\google_json'


async def translate_text(translator, text):
    """Translate text using async translator"""
    if not text:
        return ""
    
    try:
        result = await translator.translate(text, src='en', dest='de')
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return ""


async def process_file(translator, filepath):
    """Process a single JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "Files" in data:
        for file_entry in data["Files"]:
            prompt = file_entry.get("Prompt", {})
            transcript = prompt.get("Transcript", "")
            if transcript and "Translation" not in prompt:
                translation = await translate_text(translator, transcript)
                prompt["Translation"] = translation
    
    return data


async def main():
    """Main async function"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Create translator using async context manager
    async with Translator() as translator:
        for fname in os.listdir(INPUT_FOLDER):
            if not fname.lower().endswith(".json"):
                continue
            
            infile = os.path.join(INPUT_FOLDER, fname)
            try:
                augmented = await process_file(translator, infile)
                
                outfile = os.path.join(OUTPUT_FOLDER, fname)
                with open(outfile, "w", encoding="utf-8") as out_f:
                    json.dump(augmented, out_f, ensure_ascii=False, indent=2)
                
                print(f"Processed: {fname}")
            except Exception as e:
                print(f"Error processing {infile}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
