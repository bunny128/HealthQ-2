import os
import requests
import tempfile
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from modules.llm_setup import initialize_llm
from modules.file_handler import load_documents
from modules.vector_store import build_vectorstore
from modules.retriever_chain import build_conversational_rag_chain
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❗ OPENAI_API_KEY not found in environment variables")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthQRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]
    session_id: str = "default_session"

class HealthQResponse(BaseModel):
    answers: List[str]

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "HealthQ API is alive"}

@app.post("/api/v1/healthq/run", response_model=HealthQResponse)
def run_healthq(request: HealthQRequest):
    try:
        total_start = time.time()
        print("🚀 HealthQ API called")

        # Step 1: Download PDF
        t1 = time.time()
        print("📥 Downloading document...")
        response = requests.get(request.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        print(f"✅ Document downloaded in {time.time() - t1:.2f}s")

        # Step 2: Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        print("📎 Temp file saved:", tmp_path)

        # Step 3: Load documents
        t2 = time.time()
        documents = load_documents([tmp_path])
        print(f"📚 Loaded {len(documents)} documents in {time.time() - t2:.2f}s")

        # Step 4: Build vectorstore
        t3 = time.time()
        vectorstore = build_vectorstore(documents)
        print(f"📦 Vectorstore built in {time.time() - t3:.2f}s")

        # Step 5: Initialize LLM and chain
        t4 = time.time()
        llm = initialize_llm(OPENAI_API_KEY)
        rag_chain = build_conversational_rag_chain(
            llm,
            get_session_history_fn=lambda s: ChatMessageHistory(),
            filter_metadata=None
        )
        print(f"🤖 LLM + RAG chain ready in {time.time() - t4:.2f}s")

        # Step 6: Answer questions
        answers = []
        for question in request.questions:
            print("❓ Q:", question)
            t5 = time.time()
            result = rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": request.session_id}}
            )
            print(f"✅ Answered in {time.time() - t5:.2f}s")
            answers.append(result["answer"])

        print(f"🎉 Total time: {time.time() - total_start:.2f}s")
        return {"answers": answers}

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
