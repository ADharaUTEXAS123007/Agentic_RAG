from langchain.utilities import SerpAPIWrapper
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import os

# 1. Prompt for ReAct
prompt = hub.pull("hwchase17/react")
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

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# 6. Run example query
result = agent_executor.invoke({"input": "What's covered in the PDF on renewable energy?"})
print(result)

result = agent_executor.invoke({"input": "What's the latest news on AI?"})
print(result)