# 导入所需的库和模块
from collections import deque
from typing import Dict, List, Optional, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


# 任务优先级链
class TaskPrioritizationChain(LLMChain):
    """负责任务优先级排序的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        task_prioritization_template = (
            "你是一个任务优先级AI助手，负责清理格式并重新确定优先级"
            " 这是任务: {task_names}."
            " 考虑团队的最终目标: {objective}."
            " 不要删除任何任务。以编号列表形式返回结果，例如:"
            " #. First task"
            " #. Second task"
            " 从编号 {next_task_id} 开始这个任务列表."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
