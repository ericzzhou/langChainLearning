from utils import create_llm, create_output_parser, create_prompt_template


from langchain.prompts import PromptTemplate

# 创建原始模板
template = """
你是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，你能提供一个吸引人的简短描述吗？
{format_instructions}
"""

# # 根据原始模板创建Langchain提示模板
# prompt = PromptTemplate.from_template(
#     template=template,
# )

# 打印 Langchina提示目标
# print(prompt)

llm = create_llm(temperature=0.7)
# input = prompt.format(price=100, flower_name="月季花")
# output = llm.invoke(input)

# print(output)

# 导入结构化输出解释器和ResponseSchema
from langchain.output_parsers import ResponseSchema

# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(
        name="name",
        type="string",
        description="鲜花的名字",
    ),
    ResponseSchema(
        name="price",
        type="decimal",
        description="鲜花的价格",
    ),
    ResponseSchema(
        name="description",
        type="string",
        description="鲜花的描述文案",
    ),
    ResponseSchema(
        name="reason",
        type="string",
        description="为什么要写这个文案的原因和解释",
    ),
]

# 创建输出解释器
output_parser = create_output_parser(response_schemas)

# 获取格式提示
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)
# The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

# ```json
# {
#         "name": string  // 鲜花的名字
#         "price": decimal  // 鲜花的价格
#         "description": string  // 鲜花的描述文案
#         "reason": string  // 为什么要写这个文案的原因和解释
# }
# ```

# 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = create_prompt_template(template, format_instructions)
print(prompt)

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = [100, 150, 200]

for flower, price in zip(flowers, prices):
    input = prompt.format(price=price, flower_name=flower)
    output = llm.invoke(input)

    parsed_output = output_parser.parse(output)
    parsed_output["flower"] = flower
    # print(parsed_output)
