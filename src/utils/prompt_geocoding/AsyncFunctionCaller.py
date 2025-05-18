from typing import Any, Callable, Optional
import asyncio, json

class AsyncFunctionCaller:
    def __init__(
        self,
        client,
        model: str,
        system_prompt: str,
        user_prompt_template: str,
        function_schema: dict,
        function_name: str,
        parser: Callable[[dict], Any],
        rate_limiter: Any,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.function_schema = function_schema
        self.function_name = function_name
        self.parse = parser
        self.bucket = rate_limiter

    async def call(self, **prompt_kwargs) -> Any:
        # 1. rate-limit
        while not self.bucket.consume():
            await asyncio.sleep(0.1)

        # 2. format the user prompt
        user_content = self.user_prompt_template.format(**prompt_kwargs)

        # 3. invoke OpenAI
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system", "content":self.system_prompt},
                {"role":"user",   "content":user_content},
            ],
            functions=[self.function_schema],
            function_call={"name": self.function_name},
            temperature=0.0,
        )
        call = resp.choices[0].message.function_call
        raw_args = json.loads(call.arguments) if call else {}
        # 4. parse into domain object
        return self.parse(raw_args)