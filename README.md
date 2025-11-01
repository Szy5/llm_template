# LLM Template Project

一个用于快速构建 LLM 应用的 Python 工具包，提供了配置管理、LLM 模型管理、Agent 创建、搜索工具、提示词模板等常用功能的封装，让你可以快速上手开发基于 LLM 的应用。

## 功能特性

- 🔧 **配置管理**：支持 YAML 配置文件和环境变量管理
- 🤖 **多模型支持**：支持 OpenAI、Azure、Google、Dashscope、DeepSeek、Volces 等多种 LLM 提供商
- 🔍 **搜索工具**：集成 Tavily、DuckDuckGo、Brave、Arxiv、Wikipedia 等多种搜索引擎
- 💬 **Agent 构建**：基于 LangChain/LangGraph 的 Agent 创建工具
- 📝 **提示词模板**：基于 Jinja2 的提示词模板系统
- 🧠 **上下文管理**：自动管理对话上下文，支持 token 限制和消息压缩
- 🛠️ **工具函数**：JSON 修复、参数清理等实用工具

## 项目结构

```
llm_template_project/
├── src/
│   ├── config/          # 配置管理模块
│   ├── llms/            # LLM 模型管理模块
│   ├── agents/          # Agent 创建模块
│   ├── tools/           # 工具模块（搜索等）
│   ├── prompts/         # 提示词模板模块
│   ├── utils/           # 工具函数模块
│   └── graph/           # 图状态管理模块
├── config.yaml          # 配置文件
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai langchain-community langgraph
pip install python-dotenv pyyaml jinja2
pip install json-repair httpx
```

### 2. 配置环境

创建 `.env` 文件：

```bash
SEARCH_API=tavily  # 可选: tavily, duckduckgo, brave_search, arxiv, searx, wikipedia
TAVILY_API_KEY=your_tavily_api_key  # 如果使用 Tavily
BRAVE_SEARCH_API_KEY=your_brave_key  # 如果使用 Brave
```

### 3. 配置 LLM

编辑 `config.yaml`：

```yaml
BASIC_MODEL:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"
  api_key: "your-api-key"
  verify_ssl: true
  max_retries: 3
  token_limit: 8000

REASONING_MODEL:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"
  api_key: "your-api-key"
  token_limit: 8000

VISION_MODEL:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4-vision-preview"
  api_key: "your-api-key"
  token_limit: 8000

CODE_MODEL:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"
  api_key: "your-api-key"
  token_limit: 8000

SEARCH_ENGINE:
  include_raw_content: true
  include_images: true
  include_image_descriptions: true
  include_domains: []
  exclude_domains: []
```

## 核心模块使用指南

### 1. 配置管理 (`src.config`)

#### 加载配置文件

```python
from src.config import load_yaml_config

# 加载 YAML 配置文件
config = load_yaml_config("config.yaml")

# 配置文件支持环境变量替换
# 在 config.yaml 中使用 $ENV_VAR_NAME 格式
```

#### 搜索引擎配置

```python
from src.config import SELECTED_SEARCH_ENGINE, SearchEngine

# 获取当前选择的搜索引擎
print(SELECTED_SEARCH_ENGINE)  # 'tavily'

# 搜索引擎枚举类型
# SearchEngine.TAVILY
# SearchEngine.DUCKDUCKGO
# SearchEngine.BRAVE_SEARCH
# SearchEngine.ARXIV
# SearchEngine.SEARX
# SearchEngine.WIKIPEDIA
```

### 2. LLM 模型管理 (`src.llms`)

#### 获取 LLM 实例

```python
from src.llms import get_llm_by_type, get_llm_token_limit_by_type

# 获取指定类型的 LLM 实例（支持缓存）
llm = get_llm_by_type("basic")  # basic, reasoning, vision, code

# 调用 LLM
from langchain_core.messages import HumanMessage
message = HumanMessage(content="你好，介绍一下你自己")
response = llm.invoke([message])
print(response.content)

# 获取指定类型的 token 限制
token_limit = get_llm_token_limit_by_type("basic")
print(f"Token limit: {token_limit}")
```

#### 支持的 LLM 类型

- `basic`: 基础模型（默认）
- `reasoning`: 推理模型（支持思维链）
- `vision`: 视觉模型
- `code`: 代码模型

#### 支持的平台

- **OpenAI**: 通过 `ChatOpenAI`
- **Azure OpenAI**: 通过 `AzureChatOpenAI`（自动检测 `azure_endpoint`）
- **Google AI Studio**: 设置 `platform: "google_aistudio"`
- **Dashscope**: 自动检测 `dashscope.` 域名
- **DeepSeek**: 推理类型自动使用 `ChatDeepSeek`
- **Volces**: 通过 `ChatOpenAI` 配置自定义 `base_url`

### 3. Agent 创建 (`src.agents`)

#### 创建 Agent

```python
from src.agents import create_agent_
from src.tools.search import get_web_search_tool
from src.prompts import get_prompt_template
from langchain_core.messages import HumanMessage

# 创建带有搜索工具的 Agent
agent = create_agent_(
    agent_name="my_agent",
    agent_type="basic",  # 对应 config/agents.py 中的映射
    tools=[get_web_search_tool(max_search_results=3)],
    prompt_template=get_prompt_template("test")  # 从 prompts/ 目录加载模板
)

# 运行 Agent
agent_input = {
    "messages": [
        HumanMessage(content="查找与 Python 最新版本相关的信息")
    ]
}

result = agent.invoke(
    input=agent_input,
    config={"recursion_limit": 10}
)
print(result)
```

### 4. 搜索工具 (`src.tools`)

#### 使用网络搜索工具

```python
from src.tools.search import get_web_search_tool

# 创建搜索工具（根据 .env 中的 SEARCH_API 选择搜索引擎）
search_tool = get_web_search_tool(max_search_results=5)

# 在 Agent 中使用
tools = [search_tool]
```

#### 支持的搜索引擎

- **Tavily**: 高质量搜索结果，支持图片和原始内容
- **DuckDuckGo**: 免费，无需 API key
- **Brave Search**: 需要 `BRAVE_SEARCH_API_KEY`
- **Arxiv**: 学术论文搜索
- **Searx**: 元搜索引擎
- **Wikipedia**: 维基百科搜索

### 5. 提示词模板 (`src.prompts`)

#### 使用提示词模板

```python
from src.prompts import get_prompt_template, apply_prompt_template

# 方法 1: 直接获取模板内容（不包含状态变量）
template_content = get_prompt_template("test")  # 读取 prompts/test.md

# 方法 2: 应用模板（包含状态变量和当前时间）
from langgraph.prebuilt.chat_agent_executor import AgentState
state = {
    "messages": [...],
    "research_topic": "AI 研究"
}
messages_with_prompt = apply_prompt_template("test", state)
```

#### 创建提示词模板

在 `src/prompts/` 目录下创建 `.md` 文件，使用 Jinja2 语法：

```markdown
---
CURRENT_TIME: {{ CURRENT_TIME }}
---

你是一个善于查找信息的 Agent。

当前研究主题：{{ research_topic }}

请根据用户的问题，使用搜索工具查找相关信息。
```

### 6. 上下文管理 (`src.utils.context_manager`)

#### 管理对话上下文

```python
from src.utils.context_manager import ContextManager
from functools import partial

# 创建上下文管理器
token_limit = get_llm_token_limit_by_type("basic")
context_manager = ContextManager(
    token_limit=token_limit,
    preserve_prefix_message_count=3  # 保留前 3 条消息（通常是系统提示和用户输入）
)

# 作为钩子函数使用（在 Agent 调用前压缩消息）
pre_model_hook = partial(context_manager.compress_messages)

# 手动压缩消息
state = {"messages": [...]}
compressed_state = context_manager.compress_messages(state)

# 检查 token 数量
token_count = context_manager.count_tokens(messages)
print(f"Token count: {token_count}")
```

### 7. JSON 工具 (`src.utils.json_utils`)

#### JSON 修复和参数清理

```python
from src.utils.json_utils import repair_json_output, sanitize_args

# 修复可能损坏的 JSON 输出
json_string = '{"name": "test", "value": 123}'  # 可能有格式问题
repaired = repair_json_output(json_string)

# 清理工具调用参数（防止特殊字符问题）
args = '{"query": "[special] chars"}'
sanitized = sanitize_args(args)
```

### 8. 图状态管理 (`src.graph.types`)

#### 定义自定义状态

```python
from src.graph.types import State
from langchain_core.messages import HumanMessage

# State 类继承自 MessagesState，包含以下字段：
# - messages: 消息列表
# - locale: 语言环境
# - research_topic: 研究主题
# - observations: 观察结果列表
# - current_plan: 当前计划
# - final_report: 最终报告
# - enable_clarification: 是否启用澄清
# - goto: 下一个节点

state = State(
    messages=[HumanMessage(content="Hello")],
    research_topic="AI Research",
    locale="zh-CN"
)
```

## 配置 Agent 和 LLM 映射

在 `src/config/agents.py` 中定义 Agent 类型与 LLM 类型的映射：

```python
AGENT_LLM_MAP: dict[LLMType, str] = {
    "planner": "basic",
    "researcher": "basic",
    "reasoner": "reasoning"
}
```

## 完整示例

### 示例 1: 创建一个简单的搜索 Agent

```python
import os
import sys
from langchain_core.messages import HumanMessage

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import create_agent_
from src.tools.search import get_web_search_tool
from src.prompts import get_prompt_template

# 创建 Agent
agent = create_agent_(
    agent_name="search_agent",
    agent_type="basic",
    tools=[get_web_search_tool(max_search_results=3)],
    prompt_template=get_prompt_template("test")
)

# 运行
result = agent.invoke(
    input={
        "messages": [
            HumanMessage(content="查找与 LangChain 相关的信息")
        ]
    },
    config={"recursion_limit": 10}
)

print(result["messages"][-1].content)
```

### 示例 2: 使用上下文管理

```python
from src.llms import get_llm_by_type, get_llm_token_limit_by_type
from src.utils.context_manager import ContextManager
from functools import partial
from langchain_core.messages import HumanMessage, AIMessage

# 获取 LLM
llm = get_llm_by_type("basic")
token_limit = get_llm_token_limit_by_type("basic")

# 创建上下文管理器
context_manager = ContextManager(
    token_limit=token_limit,
    preserve_prefix_message_count=2
)

# 模拟一个长对话
messages = [
    HumanMessage(content="什么是 Python？"),
    AIMessage(content="Python 是一种编程语言..." * 100),  # 很长的回复
    HumanMessage(content="那它有什么特点？"),
]

# 压缩消息
state = {"messages": messages}
compressed_state = context_manager.compress_messages(state)

print(f"原始 token 数: {context_manager.count_tokens(messages)}")
print(f"压缩后 token 数: {context_manager.count_tokens(compressed_state['messages'])}")
```

## 环境变量

在 `.env` 文件中可以配置以下变量：

```bash
# 搜索引擎选择
SEARCH_API=tavily

# 搜索引擎 API Keys
TAVILY_API_KEY=your_key
BRAVE_SEARCH_API_KEY=your_key

# LLM API Keys（也可以在 config.yaml 中配置）
OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
```
