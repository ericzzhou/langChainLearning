# 少样本提示
# 在 Few-Shot 学习设置中，模型会被给予几个示例，以帮助模型理解任务，并生成正确的响应

from utils import (
    create_fewshot_prompt_template_by_samples,
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

template = "鲜花类型：{flower_type} \n场合：{occasion} \n文案：{ad_copy}"

prompt_sample_template = create_prompt_from_template(template)
# print(prompt_sample_template)
# print(prompt_sample_template.format(**samples[0]))

prompt = create_fewshot_prompt_template_by_samples(
    example_prompt_template=prompt_sample_template,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"],
    examples=samples,
)
# print(prompt)
# print(
#     prompt.format(
#         flower_type="野玫瑰",
#         occasion="爱情",
#     )
# )

modelInput = prompt.format(
    flower_type="野玫瑰",
    occasion="友情",
)

llm = create_llm()
result = llm.invoke(modelInput)
print(result)



