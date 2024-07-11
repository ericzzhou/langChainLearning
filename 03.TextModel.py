

# 调用 text 模型
from utils import create_llm


llm = create_llm()

response = llm.invoke("请给我的花店起个名")

print(response)