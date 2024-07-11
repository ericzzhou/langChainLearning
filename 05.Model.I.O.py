# 创建原始模板
from utils import create_llm, create_prompt_from_template


template = """
你是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，你能提供一个吸引人的简短描述吗？
"""

# 根据原始模板创建Langchain提示模板
prompt = create_prompt_from_template(template)
# 打印 Langchina提示目标
# print(prompt)

llm = create_llm(model="gpt-3.5-turbo-instruct")

# input = prompt.format(price=100, flower_name="月季花")
# output = llm.invoke(input)

# print(output)

flowers = ["玫瑰", "百合", "康乃馨"]
prices = [100, 150, 200]

for flower, price in zip(flowers, prices):
    input = prompt.format(price=price, flower_name=flower)
    output = llm.invoke(input)
    print(output)
