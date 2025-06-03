import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException
# from src.ocr.service import extract_text_from_image, extract_vitals

from src.ocr.service import extract_text, extract_vitals, extract_vitals_with_gpt, extract_vitals_from_in_house_model

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
