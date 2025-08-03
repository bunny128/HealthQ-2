import uuid
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def extract_section_name(text: str) -> str:
    lowered = text.lower()
    if "grace period" in lowered:
        return "Grace Period"
    elif "waiting period" in lowered:
        return "Waiting Period"
    elif "maternity" in lowered:
        return "Maternity"
    elif "cataract" in lowered:
        return "Cataract"
    elif "co-payment" in lowered or "co pay" in lowered:
        return "Co-Payment"
    elif "cumulative bonus" in lowered:
        return "No Claim Bonus"
    elif "claim" in lowered:
        return "Claims"
    elif "exclusion" in lowered:
        return "Exclusions"
    elif "ayush" in lowered:
        return "AYUSH Treatment"
    elif "hospital" in lowered:
        return "Hospital Definition"
    elif "preventive check" in lowered:
        return "Preventive Checkup"
    elif "sub-limit" in lowered or "room rent" in lowered:
        return "Room Rent Limits"
    else:
        return "General"

def infer_company(text: str) -> str:
    match = re.search(r"(.*?insurance.*?)\s+(limited|ltd\.?)", text, re.I)
    return match.group(0).strip() if match else "Unknown"

def infer_uin(text: str) -> str:
    match = re.search(r"UIN[:\s-]*([A-Z0-9]+)", text)
    return match.group(1) if match else "Unknown"

def infer_plan(text: str) -> str:
    if "plan a" in text.lower():
        return "Plan A"
    elif "plan b" in text.lower():
        return "Plan B"
    else:
        return "Standard"

def build_vectorstore(documents, source_file="unknown_file.pdf"):
    if not documents:
        raise ValueError("No documents provided to build vector store.")

    # Sample text for inference
    sample_text = documents[0].page_content if documents else ""
    company = infer_company(sample_text)
    uin = infer_uin(sample_text)

    for doc in documents:
        doc.metadata.update({
            "source": os.path.basename(source_file),
            "company": company,
            "uin": uin,
            "language": "en",
            "doc_id": str(uuid.uuid4()),
        })

    # Split and preserve metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["coverage_section"] = extract_section_name(chunk.page_content)
        chunk.metadata["plan"] = infer_plan(chunk.page_content)

    
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore
