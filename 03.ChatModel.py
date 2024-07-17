from langchain.schema import HumanMessage, SystemMessage

from utils import create_llm


chat = create_llm(is_chat_model=True)


message = [
    SystemMessage(content="你是一个很棒的智能助手."),
    HumanMessage(content="可以给我的花店起个名吗？"),
]

response = chat.invoke(message)
print(response)
