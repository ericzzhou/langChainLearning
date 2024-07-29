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


# 任务生成链
class TaskCreationChain(LLMChain):
    """负责生成任务的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        task_creation_template = (
            "你是一个使用执行代理的结果来创建任务的AI助手"
            " 创建新任务，目标如下: {objective},"
            " 最后完成的任务结果为: {result}."
            " 该结果基于以下任务描述: {task_description}."
            " 这些是未完成的任务: {incomplete_tasks}."
            " 根据结果​​，创建新的任务来完成"
            " AI助手系统不会与未完成的任务重叠。"
            " 以数组形式返回任务。"
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
