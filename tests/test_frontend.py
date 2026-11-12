"""首页必须提供可发现的知识库上传入口。"""

import asyncio
from html.parser import HTMLParser

from app.main import home


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.tags[values["id"]] = (tag, values)


def test_home_has_knowledge_selector_and_upload_button():
    page = asyncio.run(home())
    controls = Controls()
    controls.feed(page)

    assert controls.tags["knowledgeBase"][0] == "select"
    assert controls.tags["createKnowledgeBase"][0] == "form"
    assert controls.tags["uploadDocument"][0] == "form"
    assert controls.tags["documentFile"][1]["type"] == "file"
    assert controls.tags["uploadButton"][0] == "button"
    assert ".md,.txt,.pdf" == controls.tags["documentFile"][1]["accept"]
