"""可选 LangSmith 追踪；未安装 SDK 时保留原有调用行为。"""

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai
except ImportError:
    def traceable(*args, **kwargs):
        def decorate(func):
            return func
        return decorate

    def wrap_openai(client):
        return client
