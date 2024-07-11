

from utils import create_hf_model


llm =create_hf_model()

response =llm.invoke("可以给我的花店起个名吗?")


print(response)