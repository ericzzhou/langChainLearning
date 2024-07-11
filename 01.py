from utils import create_llm


llm = create_llm()
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")

print(text)
