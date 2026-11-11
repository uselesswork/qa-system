"""pytest 默认只运行本地回归，不向可观测性服务发送测试内容。"""

import os


os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
