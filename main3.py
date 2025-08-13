from langchain.utilities import SerpAPIWrapper
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tools import CustomRetrievalTool
from prompts import prompt1  # (unused here because we use hub prompt)
import os
import pickle
import hashlib
import transformers
import torch

# --- HF embeddings (CPU) ---
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# --- HF generation (GPU) ---
from langchain_community.llms import HuggingFaceHub  # not used directly, but keep if you swap later
from langchain import HuggingFacePipeline

# --------------------------
# Paths & constants
# --------------------------
PDF_PATH = "/scratch/09143/arnabd/pinn_fwi/agentic_RAG/agentic_RAG/data/Understanding_Climate_Change.pdf"
PERSIST_DIR = "./chroma_pdf_store"
HASH_FILE = os.path.join(PERSIST_DIR, "pdf_hash.txt")
MEMORY_FILE = "chat_memory.pkl"

os.makedirs(PERSIST_DIR, exist_ok=True)

# --------------------------
# Helper: file hash
# --------------------------
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------------------
# Build text-generation pipeline on GPU if available
# --------------------------
model_name_gen = "EleutherAI/gpt-neo-125m"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_gen)

# Decide device & dtype for generation
use_cuda = False
device_arg = 0 if use_cuda else -1
dtype = torch.float16 if use_cuda else torch.float32  # safe default

gen_pipe = transformers.pipeline(
    "text-generation",
    model=model_name_gen,
    tokenizer=tokenizer,
    device=device_arg,                    # <- ensures input_ids are moved correctly
    torch_dtype=dtype,
    trust_remote_code=True,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,  # avoids the pad/eos warning
)
llm = HuggingFacePipeline(pipeline=gen_pipe)

# --------------------------
# Embeddings (CPU)
# --------------------------
emb_model_name = "BAAI/bge-small-en"
emb = HuggingFaceBgeEmbeddings(
    model_name=emb_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --------------------------
# Vector store with PDF hash check
# --------------------------
def build_or_load_vectorstore(pdf_path: str, persist_dir: str, hash_file: str):
    current_hash = file_md5(pdf_path)
    rebuild = True

    if os.path.exists(persist_dir) and os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                rebuild = False
        except Exception:
            rebuild = True

    if rebuild:
        print("⚙️ Rebuilding Chroma from PDF (content changed or first run)...")
        docs = PyPDFLoader(pdf_path).load()
        vectorstore = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
        vectorstore.persist()
        with open(hash_file, "w") as f:
            f.write(current_hash)
        print("✅ Embeddings saved to:", persist_dir)
    else:
        print("🔹 Loading existing Chroma vector store...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=emb)

    return vectorstore

vectorstore = build_or_load_vectorstore(PDF_PATH, PERSIST_DIR, HASH_FILE)
retriever = vectorstore.as_retriever()

# --------------------------
# Custom retrieval tool + web tool
# --------------------------
custom_tool = CustomRetrievalTool(llm, retriever)

# SerpAPI key should be set in env SERPAPI_API_KEY
web_search = SerpAPIWrapper()
web_tool = Tool(
    name="Web_Search",
    func=web_search.run,
    description="Search the web via SerpAPI.",
)

tools = [custom_tool, web_tool]

# --------------------------
# Prompt (structured chat) from hub
# --------------------------
prompt = hub.pull("hwchase17/structured-chat-agent")

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# --------------------------
# ConversationSummaryMemory with persisted summary buffer
# --------------------------
def load_memory():
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "rb") as f:
                saved = pickle.load(f)
            chat_hist = saved.get("chat_history", [])
            summary_buf = saved.get("summary_buffer", "")
            memory.chat_memory.messages = chat_hist
            memory.moving_summary_buffer = summary_buf
            print("💾 Loaded memory: messages =", len(chat_hist), "| has summary buffer =", bool(summary_buf))
        except Exception as e:
            print("⚠️ Failed to load memory, starting fresh:", e)
    return memory

def save_memory(memory: ConversationSummaryMemory):
    try:
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump({
                "chat_history": memory.chat_memory.messages,
                "summary_buffer": getattr(memory, "moving_summary_buffer", ""),
            }, f)
    except Exception as e:
        print("⚠️ Failed to save memory:", e)

memory = load_memory()

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
)

# Seed a system message guiding tool usage
initial_system = (
    "You are an AI assistant. For every user question, you MUST use the available tools "
    "to find the answer, even if you think you know it. Do NOT answer from your own knowledge. "
    "The available tools are: custom_retrieval, Web_Search."
)
memory.chat_memory.add_message(SystemMessage(content=initial_system))
save_memory(memory)

# --------------------------
# Chat loop
# --------------------------
def main_loop():
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            save_memory(memory)
            print("👋 Bye!")
            break

        # record human message
        memory.chat_memory.add_message(HumanMessage(content=user_input))
        try:
            response = agent_executor.invoke({"input": user_input})
            output_text = response.get("output", "")
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted during invoke.")
            output_text = ""
        except Exception as e:
            print("❌ Agent error:", e)
            output_text = f"Error: {e}"

        print("AI:", output_text)

        # record AI message and persist memory (including rolling summary)
        memory.chat_memory.add_message(AIMessage(content=output_text))
        save_memory(memory)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n⏹️ KeyboardInterrupt — saving and exiting.")
        save_memory(memory)
    except Exception as e:
        print("❌ Fatal error:", e)
        save_memory(memory)
