from typing import Dict, List, Any
import json
import asyncio
from openai import AsyncOpenAI
from dmac.tool.base import BaseTool

API_KEY = "EMPTY"
MODEL = "openai/gpt-oss-120b"
BASE_URL = "http://localhost:8006/v1"


class LLM_Tool_DynamicAgent(BaseTool):

    name = "prompt_dynamic"

    description = "Execute prompt with dynamically specified agent role."

    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "agent_role": {"type": "string"},
            "conversation_messages": {"type": "array"}
        },
        "required": ["prompt"]
    }

    def __init__(self):
        super().__init__()

        self.client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

        self.fixed_system_prefix = "You are "
        self.fixed_system_suffix = (
            ". Please read the provided content (including previous conversations "
            "and the current task) and help the user complete the task or answer the question."
        )

        self.default_role_description = "a helpful assistant"

        print("[INFO] Dynamic LLM Tool initialized")

    # =============================
    # Public API
    # =============================

    def execute(self, args: Dict) -> Dict[str, Any]:
        return self._safe_run(self._execute_async(args))

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        return self._safe_run(self._batch_execute_async(args_list))

    # =============================
    # Core async logic
    # =============================

    async def _execute_async(self, args: Dict) -> Dict[str, Any]:
        try:
            messages = self._build_messages(args)
            answer = await self._call_llm(messages)
            return {"content": json.dumps({"results": [answer]}), "success": True}

        except Exception as e:
            print(f"[ERROR] DynamicAgent failed: {e}")
            return {"content": str(e), "success": False}

    async def _batch_execute_async(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        async def _single(args, idx):
            try:
                messages = self._build_messages(args)
                answer = await self._call_llm(messages)
                return {"content": json.dumps({"results": [answer]}), "success": True}
            except Exception as e:
                print(f"[ERROR] Request {idx} failed: {e}")
                return {"content": str(e), "success": False}

        tasks = [_single(args, i) for i, args in enumerate(args_list)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append({"content": str(r), "success": False})
            else:
                processed.append(r)

        return processed

    # =============================
    # Helpers
    # =============================

    def _build_messages(self, args: Dict) -> List[Dict]:

        prompt = args["prompt"]
        agent_role = args.get("agent_role")
        conversation_messages = args.get("conversation_messages", [])

        role_text = agent_role if agent_role else self.default_role_description
        print(f"[INFO] Agent role: {role_text}")

        system_prompt = (
            self.fixed_system_prefix +
            role_text +
            self.fixed_system_suffix
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        return messages

    async def _call_llm(self, messages):

        try:
            # Try chat API first
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            return response.choices[0].message.content.strip()

        except Exception:
            # Fallback to completions
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = await self.client.completions.create(
                model=MODEL,
                prompt=prompt_text,
                max_tokens=1024
            )
            return response.choices[0].text.strip()

    def _safe_run(self, coro):
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(coro)
            return loop.run_until_complete(future)
        except RuntimeError:
            return asyncio.run(coro)
