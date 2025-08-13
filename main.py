from langchain.utilities import SerpAPIWrapper
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import pickle


MEMORY_FILE = "chat_memory.pkl"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        memory = pickle.load(f)
else:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# 1. Prompt for ReAct
prompt = hub.pull("hwchase17/structured-chat-agent")
prompt

# 2. Setup LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

# 3. PDF Loader + Retriever
loader = PyPDFLoader("/scratch/09143/arnabd/pinn_fwi/agentic_RAG/agentic_RAG/data/Understanding_Climate_Change.pdf")
docs = loader.load()
emb = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, emb)
retriever = vectorstore.as_retriever()
pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

pdf_tool = Tool(
    name="PDF_Retriever",
    func=pdf_qa.run,
    description="Retrieve info from the PDF."
)

# 4. SerpAPI Web Search
search = SerpAPIWrapper()  # Requires SERPAPI_API_KEY
web_tool = Tool(
    name="Web_Search",
    func=search.run,
    description="Search the web via SerpAPI."
)

# 5. Create Agent
tools = [pdf_tool, web_tool]

agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
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
            pickle.dump(memory, f)
        # Save memory after each turn
        #with open(MEMORY_FILE, "wb") as f:
        #    pickle.dump(memory, f)
except KeyboardInterrupt:
    pass