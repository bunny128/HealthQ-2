from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_metadata_filtered_retriever(filter_metadata: dict = None):
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})

    if filter_metadata:
        retriever.search_kwargs["filter"] = filter_metadata

    return retriever

def build_conversational_rag_chain(llm, filter_metadata: dict = None):
    from modules.prompts import get_contextualize_prompt, get_qa_prompt

    retriever = get_metadata_filtered_retriever(filter_metadata)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, get_contextualize_prompt()
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, get_qa_prompt()
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return rag_chain
