# 少样本提示
# 在 Few-Shot 学习设置中，模型会被给予几个示例，以帮助模型理解任务，并生成正确的响应

from utils import (
    create_fewshot_prompt_template_by_example_selector,
    create_llm,
    create_prompt_from_template,
)

# 1. 创建一些示例
samples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。",
    },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。",
    },
    {
        "flower_type": "百合",
        "occasion": "庆祝",
        "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。",
    },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。",
    },
]

prefix = "请根据以下示例创建一个新的宣传语"
suffix = "鲜花类型: {flower_type}\n场合: {occasion}"
input_variables = ["flower_type", "occasion"]

example_prompt_template = "鲜花类型：{flower_type} \n场合：{occasion} \n文案：{ad_copy}"

prompt = create_fewshot_prompt_template_by_example_selector(
    examples=samples,
    prefix=prefix,
    suffix=suffix,
    input_variables=input_variables,
    example_prompt_template=example_prompt_template,
    example_input_variables=input_variables,
)


modelInput = prompt.format(
    flower_type="野玫瑰",
    occasion="友情",
)

llm = create_llm()
result = llm.invoke(modelInput)
print(result)
