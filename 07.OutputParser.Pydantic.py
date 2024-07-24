# 数据准备
flowers = ["玫瑰"]
prices = ["50"]


# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field

from utils import (
    create_llm,
    create_prompt_from_template,
    create_pydantic_output_parser,
)


class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


output_parser = create_pydantic_output_parser(FlowerDescription)
format_instructions = output_parser.get_format_instructions()


print(format_instructions)

# print("*******************")

prompt_template = """
你是一位专业的鲜花店文案撰写员。
对于售价为{price}元的{flower},你能提供一个吸引人的简单中文描述吗？
{format_instructions}"""

prompt = create_prompt_from_template(prompt_template, format_instructions)

# print("prompt:", prompt)

llm = create_llm()
for flower, price in zip(flowers, prices):
    input = prompt.format(
        price=price, flower=flower, format_instructions=format_instructions
    )
    print("*******************")
    print("提示词:", input)

    output = llm.invoke(input)
    print("输出:", output)

    # 解析模型的输出
    parsed_output = output_parser.parse(output)
    print("输出解析:\n", parsed_output)
