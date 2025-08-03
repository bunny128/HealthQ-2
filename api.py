import os
import requests
import tempfile
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❗ OPENAI_API_KEY not found in environment variables")

# Import internal modules
from modules.llm_setup import initialize_llm
from modules.file_handler import load_documents
from modules.vector_store import build_vectorstore
from modules.retriever_chain import build_conversational_rag_chain

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthQRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HealthQResponse(BaseModel):
    answers: List[str]

# Health check route
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "HealthQ API is alive"}

# Main processing route
@app.post("/api/v1/healthq/run", response_model=HealthQResponse)
def run_healthq(request: HealthQRequest):
    try:
        start_time = time.time()
        print("🚀 HealthQ API called")

        # Step 1: Download PDF
        print("📥 Downloading document...")
        response = requests.get(request.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        print("✅ Document downloaded.")

        # Step 2: Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        print("📎 Temp file saved:", tmp_path)

        # Step 3: Load documents
        documents = load_documents([tmp_path])
        if not documents:
            raise HTTPException(status_code=422, detail="Document content could not be extracted.")
        print(f"📚 Loaded {len(documents)} document(s).")

        # Step 4: Build vectorstore
        vectorstore = build_vectorstore(documents)
        print("📦 Vectorstore ready.")

        # Step 5: Initialize LLM and retrieval chain
        llm = initialize_llm(OPENAI_API_KEY)
        rag_chain = build_conversational_rag_chain(
            llm,
            get_session_history_fn=None,  # No session history needed
            filter_metadata=None
        )
        print("🤖 RAG pipeline initialized.")

        # Step 6: Process questions
        answers = []
        for question in request.questions:
            print(f"❓ {question}")
            result = rag_chain.invoke({"input": question})
            answers.append(result["answer"])

        print(f"✅ Completed in {time.time() - start_time:.2f} seconds.")
        return {"answers": answers}

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
