
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from pydantic import PrivateAttr
from typing import List, Dict, Any
from langchain.retrievers import BM25Retriever


# def get_chunk_by_index(vectorstore, target_index: int) -> Document:
#     """
#     Retrieve a chunk from the vectorstore based on its index in the metadata.
    
#     Args:
#     vectorstore (VectorStore): The vectorstore containing the chunks.
#     target_index (int): The index of the chunk to retrieve.
    
#     Returns:
#     Optional[Document]: The retrieved chunk as a Document object, or None if not found.
#     """
#     # This is a simplified version. In practice, you might need a more efficient method
#     # to retrieve chunks by index, depending on your vectorstore implementation.
#     all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
#     for doc in all_docs:
#         if doc.metadata.get('index') == target_index:
#             return doc
#     return None

# def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200, chunk_overlap: int = 20) -> List[str]:
#     """
#     Retrieve chunks based on a query, then fetch neighboring chunks and concatenate them, 
#     accounting for overlap and correct indexing.

#     Args:
#     vectorstore (VectorStore): The vectorstore containing the chunks.
#     retriever: The retriever object to get relevant documents.
#     query (str): The query to search for relevant chunks.
#     num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
#     chunk_size (int): The size of each chunk when originally split.
#     chunk_overlap (int): The overlap between chunks when originally split.

#     Returns:
#     List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
#     """
#     relevant_chunks = retriever.get_relevant_documents(query)
#     result_sequences = []

#     for chunk in relevant_chunks:
#         current_index = chunk.metadata.get('index')
#         if current_index is None:
#             continue

#         # Determine the range of chunks to retrieve
#         start_index = max(0, current_index - num_neighbors)
#         end_index = current_index + num_neighbors + 1  # +1 because range is exclusive at the end

#         # Retrieve all chunks in the range
#         neighbor_chunks = []
#         for i in range(start_index, end_index):
#             neighbor_chunk = get_chunk_by_index(vectorstore, i)
#             if neighbor_chunk:
#                 neighbor_chunks.append(neighbor_chunk)

#         # Sort chunks by their index to ensure correct order
#         neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

#         # Concatenate chunks, accounting for overlap
#         concatenated_text = neighbor_chunks[0].page_content
#         for i in range(1, len(neighbor_chunks)):
#             current_chunk = neighbor_chunks[i].page_content
#             overlap_start = max(0, len(concatenated_text) - chunk_overlap)
#             concatenated_text = concatenated_text[:overlap_start] + current_chunk

#         result_sequences.append(concatenated_text)

#     return result_sequences



class CustomRetrievalTool(BaseTool):
    name: str = "custom_retrieval"
    description: str = "Retrieve answers from custom document store"
    _qa_chain: any = PrivateAttr()

    def __init__(self, llm, retriever):
        super().__init__()
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,  # Pass your LLM here
            retriever=retriever,
            return_source_documents=True
        )

    def _run(self, query: str) -> str:
        print("call tool")
        result = self._qa_chain({"query": query})
        answer = result["result"]
        # Optionally, include sources
        sources = result.get("source_documents", [])
        sources_str = "\n".join([doc.page_content for doc in sources])
        return f"{answer}\n\nSources:\n{sources_str}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
    


class CustomRetrievalToolMultipleChunks(BaseTool):
    name: str = "custom_retrieval_multiple_chunks"
    description: str = "Retrieve answers from custom document store with multiple chunks and similarity scores"
    _qa_chain: any = PrivateAttr()
    _retriever: any = PrivateAttr()

    def __init__(self, llm, retriever):
        super().__init__()
        self._retriever = retriever
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def _run(self, query: str) -> str:
        print("Calling custom retrieval tool")
        
        # Get multiple chunks with similarity scores
        chunks_query_retriever = self._retriever.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve 3 most relevant chunks
        )
        
        # Get relevant documents
        relevant_docs = chunks_query_retriever.get_relevant_documents(query)
        
        # Get similarity scores (if available)
        try:
            # Try to get similarity scores from the retriever
            docs_with_scores = chunks_query_retriever.get_relevant_documents(
                query, 
                return_metadata=True
            )
        except:
            # Fallback if return_metadata is not supported
            docs_with_scores = [(doc, 0.0) for doc in relevant_docs]
        
        # Format the response with chunks and scores
        response_parts = []
        
        # Add the QA chain result
        qa_result = self._qa_chain({"query": query})
        response_parts.append(f"Answer: {qa_result['result']}")
        
        # Add chunks with similarity scores
        response_parts.append("\n\nRetrieved Chunks with Similarity Scores:")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = getattr(doc, 'metadata', {})
                page_info = metadata.get('page', 'Unknown page')
                
                response_parts.append(f"\n--- Chunk {i} (Page {page_info}, Score: {score:.4f}) ---")
                response_parts.append(f"{content[:300]}...")
            else:
                # Handle case where doc is already the content
                response_parts.append(f"\n--- Chunk {i} (Score: {score:.4f}) ---")
                response_parts.append(f"{str(doc)[:300]}...")
        
        return "\n".join(response_parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)
    

class CustomRetrievalToolHybridExpanded(BaseTool):
    name: str = "custom_retrieval_hybrid_expanded"
    description: str = (
        "Retrieve answers from a custom document store using query expansion, "
        "hybrid retrieval (semantic + keyword), and similarity scores"
    )
    _qa_chain: any = PrivateAttr()
    _retriever: any = PrivateAttr()
    _bm25_retriever: any = PrivateAttr()
    _llm: any = PrivateAttr()

    def __init__(self, llm, retriever, docs):
        """
        retriever: Vectorstore retriever
        docs: List of Document objects (for BM25 retriever)
        """
        super().__init__()
        self._retriever = retriever
        self._bm25_retriever = BM25Retriever.from_documents(docs)
        self._llm = llm
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def _expand_query(self, query: str) -> list[str]:
        """Use LLM to create semantically related queries for better recall."""
        prompt = f"""
        Expand the following search query into 3 semantically related variations 
        that might retrieve additional relevant passages from a document database.
        Query: "{query}"
        Return only the variations, one per line, no numbering.
        """
        try:
            response = self._llm.predict(prompt)
        except AttributeError:
            response = self._llm.invoke(prompt).content

        expansions = [line.strip() for line in response.split("\n") if line.strip()]
        return [query] + expansions  # include original query

    def _run(self, query: str) -> str:
        print("Calling hybrid retrieval tool with query expansion")

        expanded_queries = self._expand_query(query)
        all_docs_with_scores = []

        # 1️⃣ Semantic retrieval for each expanded query
        for q in expanded_queries:
            try:
                docs_with_scores = self._retriever.vectorstore.similarity_search_with_score(q, k=3)
                all_docs_with_scores.extend(docs_with_scores)
            except Exception as e:
                print(f"Semantic retrieval error for '{q}': {e}")

        # 2️⃣ BM25 retrieval for each expanded query
        for q in expanded_queries:
            try:
                bm25_docs = self._bm25_retriever.get_relevant_documents(q)
                # BM25 doesn't return scores in LangChain by default → assign pseudo-score
                all_docs_with_scores.extend((doc, 0.5) for doc in bm25_docs)
            except Exception as e:
                print(f"BM25 retrieval error for '{q}': {e}")

        # 3️⃣ Deduplicate by content
        seen = set()
        unique_docs_with_scores = []
        for doc, score in all_docs_with_scores:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs_with_scores.append((doc, score))

        # 4️⃣ Merge context for QA
        merged_context = " ".join(doc.page_content for doc, _ in unique_docs_with_scores)
        qa_result = self._qa_chain({"query": query, "context": merged_context})

        # 5️⃣ Format output
        response_parts = [f"Answer: {qa_result['result']}"]
        response_parts.append("\n\nRetrieved Chunks (Hybrid Retrieval) with Similarity Scores:")
        for i, (doc, score) in enumerate(unique_docs_with_scores, 1):
            page_info = doc.metadata.get('page', 'Unknown page')
            similarity_percentage = (1 - score) * 100 if score <= 1 else (1 / (1 + score)) * 100
            response_parts.append(
                f"\n--- Chunk {i} (Page {page_info}, Similarity: {similarity_percentage:.1f}%) ---"
            )
            response_parts.append(f"{doc.page_content[:300]}...")

        return "\n".join(response_parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)
    

class AnswerEvalTool(BaseTool):
    name: str = "answer_eval"
    description: str = (
        "Evaluate the quality of an answer given a question using an LLM-as-a-judge approach."
    )
    _llm: any = PrivateAttr()

    def __init__(self, llm):
        super().__init__()
        self._llm = llm

    def _run(self, input_text: str) -> str:
        """
        input_text format: 
        QUESTION: <question text>
        ANSWER: <answer text>
        """
        prompt = f"""
        You are an expert evaluator. Rate the following answer for correctness, completeness, and clarity
        on a scale of 1 (poor) to 10 (excellent). Provide reasoning for your score.

        {input_text}

        Return the result in this format:
        SCORE: <number>
        REASON: <short explanation>
        """
        try:
            return self._llm.predict(prompt)
        except AttributeError:
            return self._llm.invoke(prompt).content

    async def _arun(self, input_text: str) -> str:
        return self._run(input_text)