from typing import Literal


LLMType = Literal["basic","reasoning","vision","code"]

AGENT_LLM_MAP:dict[LLMType,str] = {
  "agent_name":"basic"
}