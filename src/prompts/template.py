import os
from jinja2 import Environment, FileSystemLoader
from langgraph.prebuilt.chat_agent_executor import AgentState

import datetime
env = Environment(
  loader = FileSystemLoader(os.path.dirname(__file__)),
  autoescape= True,
  trim_blocks= True,
  lstrip_blocks= True,
)


def get_prompt_template(prompt_name:str) -> str:
  """
  从prompts目录下获取模板文件
  Args:
    prompt_name: 模板文件名
  Returns:
    str: 模板内容
  Raises:
    ValueError: 如果模板文件不存在
  """
  try:
    template = env.get_template(f"{prompt_name}.md")
    return template.render()
  except Exception as e:
    raise ValueError(f"Error applying prompt: {prompt_name}")




def apply_prompt_template(prompt_name:str,state:AgentState) -> list:

  state_vars = {
    'CURRENT_TIME': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    **state,}

  try:
    template = env.get_template(f"{prompt_name}.md")
    system_prompt = template.render(**state_vars)
    return [{"role":"system","content":system_prompt}] +state["messages"]
  except Exception as e:
    raise ValueError(f"Error applying template {prompt_name}: {e}")

if __name__ =="__main__":
  # test 
  print(get_prompt_template("test.md"))
