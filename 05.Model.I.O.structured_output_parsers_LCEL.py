from utils import (
    create_llm,
    create_structured_output_parser,
    create_prompt_from_template,
)


from langchain.prompts import PromptTemplate

# 创建原始模板
template = """
你是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，你能提供一个吸引人的简短描述吗？
{format_instructions}
"""


llm = create_llm(temperature=0.7)

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
output_parser = create_structured_output_parser(response_schemas)

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
prompt = create_prompt_from_template(template, format_instructions)
# print(prompt)

chain = prompt | llm | output_parser
result = chain.batch(
    [
        {"flower_name": "玫瑰", "price": 100},
        {"flower_name": "百合", "price": 150},
        {"flower_name": "康乃馨", "price": 200},
    ]
)

print(result)
