"""检查本机 LangSmith 追踪配置和只读 API 连通性，不输出密钥。"""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import tracing_is_enabled


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    client = Client(timeout_ms=(3000, 3000))
    print(f"tracing_enabled={tracing_is_enabled()}")
    print(f"api_key_present={bool(client.api_key)}")
    print(f"project={os.getenv('LANGSMITH_PROJECT') or 'default'}")
    print(f"endpoint_host={urlsplit(client.api_url).hostname}")
    try:
        next(iter(client.list_projects(limit=1)), None)
    except Exception as exc:
        print(f"api_read=failed error={type(exc).__name__} status={getattr(exc, 'status_code', None)}")
        direct_session = requests.Session()
        direct_session.trust_env = False
        try:
            direct_client = Client(session=direct_session, timeout_ms=(3000, 3000))
            next(iter(direct_client.list_projects(limit=1)), None)
        except Exception as direct_exc:
            print(f"api_direct_read=failed error={type(direct_exc).__name__} status={getattr(direct_exc, 'status_code', None)}")
            raise SystemExit(1)
        else:
            print("api_direct_read=ok")
            client = direct_client
    else:
        print("api_read=ok")
    if "--show-runs" in sys.argv:
        project_name = os.getenv("LANGSMITH_PROJECT") or "default"
        try:
            project = client.read_project(project_name=project_name)
            runs = list(client.list_runs(project_id=project.id, is_root=True, limit=10,
                                         select=["id", "name", "run_type", "start_time"]))
        except Exception as exc:
            print(f"runs_read=failed error={type(exc).__name__} status={getattr(exc, 'status_code', None)}")
            raise SystemExit(1)
        print(f"project_id={project.id}")
        print(f"root_runs_found={len(runs)}")
        for run in runs:
            print(f"run name={run.name} type={run.run_type} at={run.start_time}")
    if "--show-agents" in sys.argv:
        try:
            response = client.session.get(
                f"{client.api_url}/api/v1/agents", headers={"x-api-key": client.api_key},
                params={"page_size": 20}, timeout=6,
            )
            response.raise_for_status()
            agents = response.json().get("items", [])
        except Exception as exc:
            print(f"agents_read=failed error={type(exc).__name__} status={getattr(getattr(exc, 'response', None), 'status_code', None)}")
            raise SystemExit(1)
        for agent in agents:
            print(f"agent name={agent.get('name')} id={agent.get('id')}")
            for environment in agent.get("environments", []):
                print(f"  environment={environment.get('environment')} "
                      f"tracer_session_name={environment.get('tracer_session_name')} "
                      f"tracer_session_id={environment.get('tracer_session_id')}")
    if "--smoke-trace" in sys.argv:
        @traceable(name="qa-system-trace-check", client=client)
        def probe() -> str:
            current = get_current_run_tree()
            return str(current.id) if current else "disabled"

        run_id = probe()
        if run_id == "disabled":
            print("trace_write=disabled")
            raise SystemExit(1)
        client.flush()
        for attempt in range(5):
            try:
                client.read_run(run_id)
            except Exception as exc:
                if attempt == 4:
                    print(f"trace_read=failed error={type(exc).__name__} status={getattr(exc, 'status_code', None)}")
                    raise SystemExit(1)
                else:
                    time.sleep(1)
            else:
                print("trace_read=ok")
                break


if __name__ == "__main__":
    main()
