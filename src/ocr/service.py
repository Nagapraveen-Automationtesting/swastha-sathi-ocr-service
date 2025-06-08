import os
import traceback
from pathlib import Path
import easyocr
import openai
from PIL import Image
from io import BytesIO
import numpy as np
import requests
from ollama import Client
from openai import OpenAI
from pdf2image import convert_from_bytes
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import json
import tempfile
import os
from google.cloud import storage
import mimetypes

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("D:\Workspace\Dev_Workspace\Backend_Workspace\swasthasathi-service\swasthasathi-service-230325\GCP\swasthasathi-nlp-9d6ed68ff022.json")


reader = easyocr.Reader(['en'])  # English support

def extract_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    results = reader.readtext(image_np, detail=0)
    return "\n".join(results)

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    results = reader.readtext(image_np, detail=0)
    return "\n".join(results)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes)
    all_text = []
    for img in images:
        img_np = np.array(img)
        text = reader.readtext(img_np, detail=0)
        all_text.extend(text)
    return "\n".join(all_text)

def extract_text(file_bytes: bytes, content_type: str) -> str:
    if "pdf" in content_type:
        return extract_text_from_pdf(file_bytes)
    else:
        return extract_text_from_image_bytes(file_bytes)

def extract_vitals(text: str) -> dict:
    # Placeholder for actual vitals extraction logic
    # This function should parse the text and extract relevant vitals
    # For now, it returns an empty dictionary
    # Load BioBERT model
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    # Preprocess and extract entities
    # entities = nlp(text)
    #
    # for ent in entities:
    #     print(f"Entity: {ent['word']}, Label: {ent['entity_group']}, Score: {ent['score']:.2f}")

    # Extract vitals using regex
    vital_pattern = re.findall(
        r"(?P<param>[A-Za-z ]+):?\s*(?P<value>\d+(\.\d+)?(?:/\d+)?)[\s]*?(?P<unit>mg/dL|g/dL|mmHg|/mm3|%|bpm|°F)?",
        text
    )

    # Format result
    output = {
        "Hospital Name": "XYZ",
        "Report Date": "24-05-2025",
        "Report Number": 12345,
        "Vitals": []
    }

    for param, value, _, unit in vital_pattern:
        output["Vitals"].append({
            "Param Name": param.strip(),
            "Unit": unit or "",
            "Value": value,
            "Norma Range": ""
        })

    # Print formatted JSON
    print(json.dumps(output, indent=2))

    return output

openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
def extract_vitals_with_gpt(report_text: str) -> dict:
    prompt = f"""
        You are a medical AI assistant. Extract vitals from the following diagnostic report text and return JSON in the format:
        
        {{
          "Hospital Name": "XYZ",
          "Report Date": "24-05-2025",
          "Report Number": 12345,
          "Vitals": [
            {{
              "Param Name": "Hemoglobin",
              "Unit": "g/dL",
              "Value": "13.5",
              "Norma Range": ""
            }},
            ...
          ]
        }}
        
        Report Text:
        \"\"\"
        {report_text}
        \"\"\"
        """

    try:

        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in extracting vitals from diagnostic reports."},
                {"role": "user", "content": report_text}
            ]
        )
        output = response['choices'][0]['message']['content']
        print("GPT-4 Response:\n", output)
        return output

    except Exception as e:
        print("OpenAI API Error:", e)
        return {}

def extract_vitals_from_in_house_model(text: str):
    try:
        prompt = f"""
            You are a medical AI assistant. Extract only medical vitals from the following text and return a JSON object with this structure:

            {{
              "Hospital Name": "<string or empty if not present>",
              "Report Date": "<DD-MM-YYYY or empty if not present>",
              "Report Number": "<string or empty if not present>",
              "Vitals": [
                {{
                  "Param Name": "<vital name>",
                  "Unit": "<unit or empty if not present>",
                  "Value": "<value>",
                  "Normal Range": "<range or empty if not present>"
                }}
              ]
            }}

            Text:
            {text}
            """
        payload = {
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload,
                                 headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        vitals_data = data.get("response")
        if isinstance(vitals_data, str):
            return json.loads(vitals_data)

        print(f"Model Response: {vitals_data}")
        return vitals_data
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

async def download_and_process_temp_file(file_name: str):
    # Initialize client
    storage_client = storage.Client()
    print(f"\n\n\n\nDownloading file: {file_name}\n\n\n\n")
    # Get the bucket and blob
    bucket = storage_client.bucket("ss-ocr-new")
    blob = bucket.blob(file_name)
    temp_dir = Path("./TEMP").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / Path(file_name).name

    # Download blob to temporary file
    blob.download_to_filename(temp_file_path)
    print("Download complete.")

    try:
        # --- Your processing logic here ---
        with open(temp_file_path, 'rb') as f:
            contents = f.read()
            content_type, _ = mimetypes.guess_type(str(temp_file_path))
            if content_type is None:
                content_type = "application/octet-stream"
            text = extract_text(contents, content_type)
            print(f"\n\n**********************{content_type}********************************")
            print(f"Extracted text : {text}")

            print(f"******************************************************\n\n")
            return  text
    finally:
        # Clean up: Delete the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted.")
