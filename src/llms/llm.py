
import os
from pathlib import Path
from typing import Any, Dict, get_args

import httpx

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.config import load_yaml_config
from src.config.agents import LLMType
from src.llms.providers.dashscope import ChatDashscope


def _get_config_file_path() -> str:
  return str((Path(__file__).parent.parent.parent/"config.yaml").resolve())


_llm_cache: dict[LLMType, BaseChatModel] = {}



def get_llm_by_type(llm_type:LLMType) -> BaseChatModel:

  if llm_type in _llm_cache:
    return _llm_cache[llm_type]

  conf = load_yaml_config(_get_config_file_path())
  llm =_create_llm_use_conf(llm_type,conf)
  _llm_cache[llm_type] = llm
  return llm

def get_llm_token_limit_by_type(llm_type: str) -> int:
    """
    通过config.yaml获取指定type的LLM的token限制

    Args:
        llm_type (str): The type of LLM.

    Returns:
        int: The maximum token limit for the specified LLM type.
    """

    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    conf = load_yaml_config(_get_config_file_path())
    llm_max_token = conf.get(config_key, {}).get("token_limit")
    return llm_max_token


def _get_llm_type_config_keys() -> dict[str, str]:
    """获得LLM类型与配置文件中的键的映射"""

    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
    }




def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel:

    """
    Create LLM instance using config.yaml：
    1. 目前支持创建Volces、Google、Azure、Dashscope、DeepSeek、ChatOpenAI的LLM实例
    2. ChatOpenAI需要配置的key：base_url，model，api_key，verify_ssl，max_retries
    
    其他的等使用的时候待补充
    """
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    if not config_key:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    llm_conf = conf.get(config_key, {})
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM configuration for {llm_type}: {llm_conf}")

    # Get configuration from environment variables
    # env_conf = _get_env_llm_conf(llm_type)

    # Merge configurations, with environment variables taking precedence
    # merged_conf = {**llm_conf, **env_conf}

    merged_conf = {**llm_conf}

    # Remove unnecessary parameters when initializing the llm client
    if "token_limit" in merged_conf:
        merged_conf.pop("token_limit")

    if not merged_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Add max_retries to handle rate limit errors
    if "max_retries" not in merged_conf:
        merged_conf["max_retries"] = 3

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # 不需要校验证书的情况：Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client

    # Check if it's Google AI Studio platform based on configuration
    platform = merged_conf.get("platform", "").lower()
    is_google_aistudio = platform == "google_aistudio" or platform == "google-aistudio"
    # google ai studio平台
    if is_google_aistudio:
        # Handle Google AI Studio specific configuration
        gemini_conf = merged_conf.copy()

        # Map common keys to Google AI Studio specific keys
        if "api_key" in gemini_conf:
            gemini_conf["google_api_key"] = gemini_conf.pop("api_key")

        # Remove base_url and platform since Google AI Studio doesn't use them
        gemini_conf.pop("base_url", None)
        gemini_conf.pop("platform", None)

        # Remove unsupported parameters for Google AI Studio
        gemini_conf.pop("http_client", None)
        gemini_conf.pop("http_async_client", None)

        return ChatGoogleGenerativeAI(**gemini_conf)
    # azure 平台
    if "azure_endpoint" in merged_conf or os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(**merged_conf)

    # Check if base_url is dashscope endpoint
    if "base_url" in merged_conf and "dashscope." in merged_conf["base_url"]:
        if llm_type == "reasoning":
            merged_conf["extra_body"] = {"enable_thinking": True}
        else:
            merged_conf["extra_body"] = {"enable_thinking": False}
        return ChatDashscope(**merged_conf)

    if llm_type == "reasoning":
        merged_conf["api_base"] = merged_conf.pop("base_url", None)
        return ChatDeepSeek(**merged_conf)
    # ChatOpenAI 平台创建LLM实例
    else:
        return ChatOpenAI(**merged_conf)




if __name__ =="__main__":
  # test method get_llm_by_type
  llm = get_llm_by_type("basic")
  message = HumanMessage(content="Hello,who are you?")
  response = llm.invoke(input=[message])
  print(response.content)

  #test method get_llm_token_limit_by_type
  token_limit = get_llm_token_limit_by_type("basic")
  print(f"The token limit for basic LLM is: {token_limit}")


