import os
import requests
import tempfile
import time  # ✅ For logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from modules.file_handler import load_documents
from modules.vector_store import build_vectorstore
from langchain_community.chat_message_histories import ChatMessageHistory
from modules.retriever_chain import build_conversational_rag_chain
from modules.llm_setup import initialize_llm

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Init app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
def run_hackrx(request: HackRxRequest):
    try:
        total_start = time.time()
        print("🔹 API called")
        
        # Step 1: Download PDF
        t1 = time.time()
        print("📥 Downloading document...")
        response = requests.get(request.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document")
        print(f"✅ Document downloaded in {time.time() - t1:.2f}s")

        # Save to temp file
        t2 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        print(f"📝 Temp file written in {time.time() - t2:.2f}s")

        # Step 2: Load PDF
        t3 = time.time()
        print("📚 Loading document...")
        documents = load_documents([tmp_path])
        print(f"✅ Document loaded in {time.time() - t3:.2f}s")

        # Step 3: Build vectorstore
        t4 = time.time()
        print("📦 Building vectorstore...")
        vectorstore = build_vectorstore(documents)
        retriever = vectorstore.as_retriever()
        print(f"✅ Vectorstore built in {time.time() - t4:.2f}s")

        # Step 4: Init LLM and RAG
        t5 = time.time()
        print("🤖 Initializing LLM and RAG chain...")
        llm = initialize_llm(OPENAI_API_KEY)
        rag_chain = build_conversational_rag_chain(llm, retriever, lambda _: ChatMessageHistory())
        print(f"✅ LLM + RAG initialized in {time.time() - t5:.2f}s")

        # Step 5: Run inference
        answers = []
        for question in request.questions:
            t_question = time.time()
            print(f"❓ Answering: {question}")
            result = rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "hackrx"}})
            print(f"✅ Answered in {time.time() - t_question:.2f}s")
            answers.append(result["answer"])

        print(f"✅ Total time taken: {time.time() - total_start:.2f}s")
        return {"answers": answers}

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)