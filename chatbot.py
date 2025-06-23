import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import QA_PROMPT_TEMPLATE
from dotenv import load_dotenv
from typing import Any
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=QA_PROMPT_TEMPLATE
)
llm = ChatOpenAI(
    model="gpt-4.1", 
    temperature=0.3,
    max_tokens=256,
    openai_api_key=OPENAI_API_KEY
)
chain = prompt | llm | StrOutputParser()

async def answer_question(pinecone_index: Any, question: str, k: int = 3) -> str:
    print(f"Starting question processing: '{question[:50]}...' at {time.strftime('%H:%M:%S')}")
    total_start_time = time.time()
    
    # Document retrieval timing
    print(f"Retrieving top {k} documents...")
    retrieval_start_time = time.time()
    
    # Compute query embedding and search via Pinecone
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    query_vec = embedder.embed_query(question)

    res = pinecone_index.query(vector=query_vec, top_k=k, include_metadata=True)
    matches = res.matches if hasattr(res, "matches") else res["matches"]
    retrieval_time = time.time() - retrieval_start_time
    print(f"Document retrieval completed in {retrieval_time:.2f} seconds")
    print(f"Retrieved {len(matches)} documents")
    
    # Prepare context
    context_start_time = time.time()
    context_chunks = []
    for m in matches:
        meta = m.metadata or {}
        text = meta.get("text") or meta.get("chunk_text") or ""
        context_chunks.append(text)

    context = "\n\n".join(context_chunks)
    context_time = time.time() - context_start_time
    print(f"Context prepared in {context_time:.3f} seconds (length: {len(context)} chars)")
    
    # LLM inference timing
    print("Generating answer with LLM...")
    llm_start_time = time.time()
    result: str = await chain.ainvoke({"question": question, "context": context})
    llm_time = time.time() - llm_start_time
    print(f"LLM response generated in {llm_time:.2f} seconds")
    print(f"Answer length: {len(result)} characters")
    
    # Total timing
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Question answered successfully at {time.strftime('%H:%M:%S')}")
    print("-" * 60)
    
    return result
