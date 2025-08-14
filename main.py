from langchain.utilities import SerpAPIWrapper
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tools import CustomRetrievalTool
from prompts import written_react_prompt
import os
import pickle
import dill
#from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain import HuggingFacePipeline
#from transformers import pipeline
import transformers
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from huggingface_hub import InferenceClient

# Create a pipeline (local or via HF Hub)

#prompt = prompt1
PERSIST_DIR = "./chroma_pdf_store"

# model = "Qwen/Qwen2.5-Coder-32B-Instruct"
# tokenizer = transformers.AutoTokenizer.from_pretrained(model)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.eos_token_id,
#     device="cpu"
# )
# llm = HuggingFacePipeline(pipeline=pipeline)
llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_tokens=1000)


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
emb = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Wrap the pipeline for LangChain
#llm = HuggingFacePipeline(pipeline=pipe, temperature=0.0)


MEMORY_FILE = "chat_memory.pkl"
# if os.path.exists(MEMORY_FILE):
#     with open(MEMORY_FILE, "rb") as f:
#         saved_data = pickle.load(f)
#         print("saved_data :", saved_data)
#     memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
#     memory.chat_memory.messages = saved_data["chat_history"]
#     #memory.moving_summary_buffer = saved_data.get("summary_buffer", "")
# else:
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
# 1. Prompt for ReAct
#prompt = hub.pull("hwchase17/structured-chat-agent")
#prompt
#from prompts import simple_react_prompt#
prompt = written_react_prompt
#prompt = hub.pull("hwchase17/react")

# 3. PDF Loader + Retriever
loader = PyPDFLoader("/scratch/09143/arnabd/pinn_fwi/agentic_RAG/agentic_RAG/data/Understanding_Climate_Change.pdf")
docs = loader.load()
#emb = OpenAIEmbeddings()


if os.path.exists(PERSIST_DIR):
    print("🔹 Loading existing Chroma vector store...")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
else:
    print("🔹 Creating Chroma vector store from PDF...")
    loader = PyPDFLoader("/scratch/09143/arnabd/pinn_fwi/agentic_RAG/agentic_RAG/data/Understanding_Climate_Change.pdf")
    docs = loader.load()
    vectorstore = Chroma.from_documents(docs, emb, persist_directory=PERSIST_DIR)
    vectorstore.persist()
    print("✅ Embeddings saved to", PERSIST_DIR)

retriever = vectorstore.as_retriever()
custom_tool = CustomRetrievalTool(llm, retriever)

# pdf_tool = Tool(
#     name="PDF_Retriever",
#     func=pdf_qa.run,
#     description="Retrieve info from the PDF."
# )

# 4. SerpAPI Web Search
search = SerpAPIWrapper()  # Requires SERPAPI_API_KEY
web_tool = Tool(
    name="Web_Search",
    func=search.run,
    description="Search the web via SerpAPI."
)

# 5. Create Agent
tools = [custom_tool]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = ("You are an AI assistant. For every user question, you MUST use the available tools to find the answer, "
    "even if you think you know the answer. Do NOT answer from your own knowledge. "
    "The available tools are: custom_retrieval, Web_Search."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# --- Chat Loop ---
try:
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            # Add the user's message to the conversation memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))
        response = agent_executor.invoke({"input": user_input})
        print("AI:", response["output"])

        # Add the agent's response to the conversation memory
        memory.chat_memory.add_message(AIMessage(content=response["output"]))
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump({"chat_history": memory.chat_memory.messages}, f)
        # Save memory after each turn
        #with open(MEMORY_FILE, "wb") as f:
        #    pickle.dump(memory, f)
except KeyboardInterrupt:
    pass