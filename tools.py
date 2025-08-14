
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from pydantic import PrivateAttr
from typing import List, Dict, Any



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
    

class CustomRetrievalToolMultipleChunksScore(BaseTool):
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
        print("Calling custom retrieval tool with similarity scores")
        
        # Use similarity_search_with_score to get documents with scores
        try:
            # Get documents with similarity scores
            docs_with_scores = self._retriever.vectorstore.similarity_search_with_score(
                query, 
                k=3
            )
            
            # Get the QA chain result
            qa_result = self._qa_chain({"query": query})
            
            # Format the response
            response_parts = []
            response_parts.append(f"Answer: {qa_result['result']}")
            response_parts.append("\n\nRetrieved Chunks with Similarity Scores:")
            
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                content = doc.page_content
                metadata = doc.metadata
                page_info = metadata.get('page', 'Unknown page')
                
                # Convert similarity score to a more readable format
                # Higher score = more similar (cosine similarity)
                similarity_percentage = (1 - score) * 100 if score <= 1 else (1 / (1 + score)) * 100
                
                response_parts.append(f"\n--- Chunk {i} (Page {page_info}, Similarity: {similarity_percentage:.1f}%) ---")
                response_parts.append(f"{content[:300]}...")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            print(f"Error getting similarity scores: {e}")
            # Fallback to regular retrieval
            relevant_docs = self._retriever.get_relevant_documents(query)
            qa_result = self._qa_chain({"query": query})
            
            response_parts = []
            response_parts.append(f"Answer: {qa_result['result']}")
            response_parts.append("\n\nRetrieved Chunks (similarity scores not available):")
            
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.page_content
                metadata = doc.metadata
                page_info = metadata.get('page', 'Unknown page')
                
                response_parts.append(f"\n--- Chunk {i} (Page {page_info}) ---")
                response_parts.append(f"{content[:300]}...")
            
            return "\n".join(response_parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)
