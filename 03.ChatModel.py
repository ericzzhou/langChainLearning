
from langchain.schema import HumanMessage,SystemMessage

from utils import create_chat_model


chat = create_chat_model()


message = [
    SystemMessage( content= "你是一个很棒的智能助手."),
    HumanMessage( content= "可以给我的花店起个名吗？" )
]

response = chat.invoke(message)
print(response)