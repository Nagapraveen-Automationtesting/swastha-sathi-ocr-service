import traceback
import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
# from src.ocr.service import extract_text_from_image, extract_vitals

from src.ocr.service import extract_text, extract_vitals, extract_vitals_with_gpt, extract_vitals_from_in_house_model, \
    download_and_process_temp_file

router = APIRouter()

# @router.post("/upload-report")
# async def upload_report(report: UploadFile = File(...)):
#     try:
#         contents = await report.read()
#         text = extract_text_from_image(contents)
#         return { "success": True, "extracted_text": text }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-report")
async def upload_report(report: UploadFile = File(...)):
        try:
            contents = await report.read()
            text = extract_text(contents, report.content_type)
            print(f"Extracted text: {text}")
            # extracted_vitals = extract_vitals(text)
            extracted_vitals = extract_vitals_from_in_house_model(text)
            return { "success": True, "extracted_vitals": extracted_vitals }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-vitals")
async def extract_vitals(path: str):
        try:
            start_time = datetime.datetime.now()
            text = await download_and_process_temp_file(path)
            # contents = await report.read()
            # text = extract_text(contents, report.content_type)
            # print(f"Extracted text: {text}")
            # # extracted_vitals = extract_vitals(text)
            end_time = datetime.datetime.now()
            print(f"Time taken to download and process file: {end_time - start_time}")
            start_q_time =datetime.datetime.now()
            extracted_vitals = extract_vitals_from_in_house_model(text)
            end_q_time = datetime.datetime.now()
            print(f"Time taken to extract vitals: {end_q_time - start_q_time}")
            print(f"Total Time taken to extract vitals: {end_q_time - start_time}")

            return { "success": True, "extracted_vitals": extracted_vitals }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
