# api.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
from acualfiletorun import *


load_dotenv()

app = FastAPI()

# Allow Salesforce to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Salesforce domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Upload PDF Endpoint ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Your existing function
        from app import load_and_index_pdf
        class FakeFile:
            name = file_path
        result = load_and_index_pdf(FakeFile())
        return {"status": "success", "message": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Ask Question Endpoint ---
@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")

        from app import generate_answer
        answer = generate_answer(question)
        return {"status": "success", "answer": answer}

    except Exception as e:
        return {"status": "error", "answer": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )