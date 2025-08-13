from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from langchain.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=(
        "You are an AI assistant. For every user question, you MUST use the available tools to find the answer, "
        "even if you think you know the answer. Do NOT answer from your own knowledge.\n\n"
        "Available tools:\n{tools}\n\n"
        "Tool names: {tool_names}\n\n"
        "When you use a tool, always show your reasoning step by step.\n\n"
        "User: {input}\n\n"
        "{agent_scratchpad}\n"
        "(Reminder: Always respond in a JSON blob, and always use a tool to answer.)"
    )
)

prompt1 = (
    "Respond to the human as helpfully and accurately as possible. "
    "You have access to the following tools: {tools} "
    "Use a JSON blob to specify a tool by providing an 'action' key (tool name) "
    "and an 'action_input' key (tool input). "
    "Valid 'action' values: 'Final Answer' or {tool_names}. "
    "Provide only ONE action per JSON blob, as shown:\n\n"
    "```\n"
    "{\n"
    '  "action": $TOOL_NAME,\n'
    '  "action_input": $INPUT\n'
    "}\n"
    "```\n\n"
    "Follow this format:\n\n"
    "Question: input question to answer\n"
    "Thought: consider previous and subsequent steps\n"
    "Action:\n"
    "```\n"
    "$JSON_BLOB\n"
    "```\n"
    "Observation: action result\n"
    "... (repeat Thought/Action/Observation N times)\n"
    "Thought: I know what to respond\n"
    "Action:\n"
    "```\n"
    "{\n"
    '  "action": "Final Answer",\n'
    '  "action_input": "Final response to human"\n'
    "}\n"
    "```\n\n"
    "Begin! Reminder to ALWAYS respond with a valid JSON blob of a single action. "
    "Use tools if necessary. Respond directly if appropriate. "
    "Format is Action:```$JSON_BLOB``` then Observation:.\n"
    "Thought:"
)