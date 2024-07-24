# 自动修复解析器（OutputFixingParser）实战
# 数据准备
flowers = ["玫瑰"]
prices = ["50"]


# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
from typing import List
from utils import (
    create_llm,
    create_prompt_from_template,
    create_pydantic_output_parser,
)

from langchain.output_parsers import OutputFixingParser

# 使用Pydantic创建一个数据格式，表示花
class Flower(BaseModel):
    name: str = Field(description="name of a flower")
    colors: List[str] = Field(description="the color of this flower")


# 定义一个用于获取某种花的颜色列表的查询
flower_query = "Generate the charaters for a random flower."

# 定义一个格式不正确的输出
misformatted = "{'name':'康乃馨','colors':['粉红色','白色','紫色','黄色']}"
# misformatted = '{"name":"康乃馨","colors":["粉红色","白色","紫色","黄色"]}'

# 创建一个用于解析输出的 pydantic 解析器，此处希望解析为 Flower 格式
parser = create_pydantic_output_parser(pydantic_object=Flower)
# output = parser.parse(misformatted)
# print(output)

new_parser = OutputFixingParser.from_llm(parser=parser, llm=create_llm())
output = new_parser.parse(misformatted)
print(output)

# 结论：
# 运行失败，按照教程说明，使用一个错误的json格式不会再报错。
# 实际上运行结果依然报错