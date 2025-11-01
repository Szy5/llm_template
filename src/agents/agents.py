from functools import partial
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from src.tools.search import get_web_search_tool
from src.utils.context_manager import ContextManager
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.prompts import get_prompt_template,apply_prompt_template


def create_agent_(
    agent_name: str,
    agent_type: str,
    tools: list,
    prompt_template: str,
):
    return create_agent(
    name=agent_name,
    model=get_llm_by_type(AGENT_LLM_MAP[agent_type]),
    tools=tools,
    system_prompt=prompt_template,
    )

if __name__ == "__main__":
  import os
  import sys
  project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  if project_root not in sys.path:
    sys.path.insert(0, project_root)
  from src.prompts import template
  from jinja2 import Environment, FileSystemLoader, select_autoescape
  template_module = os.path.join(project_root, "src", "prompts", "template.py")
  if os.path.exists(template_module):
    template.env = Environment(
      loader=FileSystemLoader(os.path.join(project_root, "src", "prompts")),
      autoescape=select_autoescape(),
      trim_blocks=True,
      lstrip_blocks=True,
    )
  from src.prompts import get_prompt_template


  # llm_token_limit = get_llm_token_limit_by_type("basic")
  # pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
  

  # 调用 tools的agent
  message = get_prompt_template("test")
  
  graph = create_agent_(
    agent_name = "test_agent",
    agent_type = "planner",
    tools = [get_web_search_tool(3)],
    prompt_template = message,
  )
  
  agent_input = {
      "messages": [
          HumanMessage(content="查找与猫眼公司相关的信息"),
      ]
  }
  result = graph.invoke(input = agent_input,
  config={"recursion_limit": 10}
  )
  print(result)

  
  

