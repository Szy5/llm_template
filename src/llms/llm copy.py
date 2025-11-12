import os
from pathlib import Path
from typing import Any, Dict

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_openai import ChatOpenAI
from src.config.agents import LLMType
from src.config import load_yaml_config




def get_config_file_path() -> str:
  return str((Path(__file__).parent.parent.parent/"config.yaml").resolve())

def _get_llm_type_config_keys() -> dict[str, str]:
    """获得LLM类型与配置文件中的键的映射"""

    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
        "react_fastchat": "REACT_FASTCHAT_MODEL",
    }

_llm_chache:dict[LLMType,BaseChatModel] = {}


def get_llm_by_type(llm_type:LLMType) -> BaseChatModel:
  if llm_type in _llm_chache:
    return _llm_chache[llm_type]
  
  yaml_conf = load_yaml_config(get_config_file_path())
  llm = create_llm_use_conf(llm_type,yaml_conf)
  _llm_chache[llm_type] = llm
  return llm


def create_llm_use_conf(llm_type:LLMType,conf:Dict[str,Any]) -> BaseChatModel:
  llm_type_conf_keys = _get_llm_type_config_keys()
  config_key = llm_type_conf_keys.get(llm_type)
  if not config_key:
    raise ValueError(f"Unknown LLM Type: {llm_type} ,Please add it")
  
  llm_conf = conf.get(config_key,{})
  if not isinstance(llm_conf,dict):
    raise ValueError(f"Invalid LLM Configuration for {llm_type}: {llm_conf}")

  merged_conf = {**llm_conf}
  # 最大重试次数
  if "max_retries" not in merged_conf:
    merged_conf["max_retries"] = 5
  # 是否校验SSL
  verifiy_ssl = merged_conf.pop("verify_ssl",True)
  if not verifiy_ssl:
    http_client = httpx.client(verify=False)
    http_async_client = httpx.AsyncClient(verify=False)
    merged_conf["http_client"] = http_client
    merged_conf["http_async_client"] = http_async_client    

  if llm_type =="react_fastchat":
    return ChatOpenAI(**merged_conf)


if __name__ =="__main__":
  # test method get_llm_by_type
  llm = get_llm_by_type("react_fastchat")
  message = HumanMessage(content="Hello,who are you?")
  response = llm.invoke(input=[message])
  print(response.content)

