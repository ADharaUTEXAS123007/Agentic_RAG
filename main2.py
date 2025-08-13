from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
import pickle
import os

# --- Load or create persistent memory ---
MEMORY_FILE = "chat_memory.pkl"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        memory = pickle.load(f)
else:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- SerpAPI Tool ---
search = SerpAPIWrapper()
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Useful for answering questions about current events or factual topics."
)

# --- PDF RAG Tool ---
loader = PyPDFLoader("/scratch/09143/arnabd/agentic_RAG/data/Understanding_Climate_Change.pdf")
docs = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
pdf_tool = Tool(
    name="PDF QA",
    func=rag_chain.run,
    description="Useful for answering questions based on the contents of the PDF."
)

# --- Create ReAct Agent ---
tools = [search_tool, pdf_tool]
prompt = None  # create_react_agent uses its own default prompt
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# --- Chat Loop ---
try:
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent_executor.invoke({"input": user_input})
        print("AI:", response["output"])
        # Save memory after each turn
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump(memory, f)
except KeyboardInterrupt:
    pass