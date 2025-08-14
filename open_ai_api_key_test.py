from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import os
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What should we do to stop pollution?"),]
llm = ChatOpenAI(openai_api_key=os.getenv("MODEL_GATEWAY_API_KEY"), model="avathon-openai/o4-mini", temperature=1,base_url=os.getenv("MODEL_GATEWAY_URL"))
print(llm.invoke(messages))