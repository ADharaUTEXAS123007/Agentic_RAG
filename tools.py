
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from pydantic import PrivateAttr



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
        sources_str = "\n".join([doc.page_content[:100] for doc in sources])
        return f"{answer}\n\nSources:\n{sources_str}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
