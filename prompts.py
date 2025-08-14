from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,)
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


# Simple version of the structured chat agent prompt
simple_structured_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful AI assistant with access to these tools:\n\n{tools}\n\n"
        "Use tools by providing a JSON blob with 'action' (tool name) and 'action_input' (tool input).\n\n"
        "Valid actions: 'Final Answer' or {tool_names}\n\n"
        "Format:\n"
        "Question: user question\n"
        "Thought: your reasoning\n"
        "Action: JSON blob\n"
        "Observation: tool result\n"
        "Repeat until you have the answer, then use 'Final Answer'"
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(
        "{input}\n\n{agent_scratchpad}"
    )
])





# Simple ReAct prompt
simple_react_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to these tools:

{tools}

IMPORTANT: You MUST follow this EXACT format:

Question: {input}
Thought: {agent_scratchpad}
Action: [choose from {tool_names}]
Action Input: [provide the input for the tool]

Example:
Question: What is the weather like?
Thought: I need to search for current weather information
Action: Web_Search
Action Input: current weather forecast

Begin now!"""
)


# Simple ReAct prompt
written_react_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to these tools:

{tools}

Use the following format to answer the question:

Question: the input question you must answer
Thought: you should always think about what to do do
Action: the action to take, should be one of  [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Example:
Question: What is the weather like?
Thought: I need to search for current weather information
Action: Web_Search
Action Input: current weather forecast

Begin!
Question: {input}
Thought: {agent_scratchpad}"""
)