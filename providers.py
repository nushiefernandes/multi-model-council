"""
Model provider integrations for Council v2.

WHAT'S DIFFERENT FROM V1:
------------------------
The original providers.py only had:
    - complete() -> returns raw text
    - stream() -> yields text chunks

This version adds:
    - complete_structured() -> returns validated Pydantic model

WHY STRUCTURED OUTPUTS MATTER:
-----------------------------
Without structured outputs:
    response = await provider.complete("Propose an approach...")
    # response["content"] = "I think we should use JWT tokens because..."
    # Now you have to PARSE this text to extract approach, rationale, etc.
    # This is brittle and error-prone

With structured outputs:
    response = await provider.complete_structured(messages, schema=Proposal)
    # response = Proposal(approach="Use JWT", rationale="...", effort="medium", ...)
    # Already parsed, validated, typed!

HOW EACH PROVIDER IMPLEMENTS IT:
-------------------------------
1. Anthropic (Claude): Uses "tool use" - we define a tool with the schema,
   Claude "calls" the tool with structured data

2. OpenAI (GPT-4): Uses response_format with json_schema - native support

3. Ollama (DeepSeek): Uses prompt engineering + JSON mode - less reliable,
   needs retry logic
"""

import os
import json
import subprocess
import time
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, TypeVar, Type
import httpx
from pydantic import BaseModel, ValidationError

# Import our schemas
from schemas import get_json_schema

# Type variable for generic schema returns
T = TypeVar('T', bound=BaseModel)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_ollama_running(model: str = None, base_url: str = "http://localhost:11434") -> bool:
    """
    Ensure Ollama is running and optionally pull the model.
    Returns True if Ollama is ready, False otherwise.

    This was added in v1 to auto-start Ollama when needed.
    """
    import urllib.request
    import urllib.error

    # Check if Ollama is already running
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        print("✓ Ollama is running")
    except (urllib.error.URLError, ConnectionRefusedError):
        print("Starting Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait for it to be ready
        for _ in range(30):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
                print("✓ Ollama started")
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                continue
        else:
            print("✗ Failed to start Ollama")
            return False

    # Check if model is available, pull if needed
    if model:
        try:
            resp = urllib.request.urlopen(f"{base_url}/api/tags", timeout=10)
            data = json.loads(resp.read().decode())
            available_models = [m["name"].split(":")[0] for m in data.get("models", [])]
            model_base = model.split(":")[0]

            if model_base not in available_models:
                print(f"Pulling {model}... (this may take a while)")
                result = subprocess.run(["ollama", "pull", model], capture_output=False)
                if result.returncode == 0:
                    print(f"✓ Model {model} ready")
                else:
                    print(f"✗ Failed to pull {model}")
                    return False
            else:
                print(f"✓ Model {model} available")
        except Exception as e:
            print(f"Warning: Could not check model availability: {e}")

    return True


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that might have markdown code blocks or extra content.

    Models sometimes return:
        Here's the JSON:
        ```json
        {"approach": "..."}
        ```

    This extracts just the JSON part.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find raw JSON (starts with { or [)
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if json_match:
        return json_match.group(1)

    # Return original text and let JSON parser fail with good error
    return text


def make_openai_compatible_schema(schema: dict) -> dict:
    """
    Make a JSON schema compatible with OpenAI's strict mode.

    OpenAI requires:
    1. additionalProperties: false on all objects
    2. All properties must be required (for strict mode)

    This recursively fixes the schema.
    """
    schema = schema.copy()

    # Remove unsupported fields
    schema.pop("description", None)  # Top-level description not supported
    schema.pop("examples", None)

    def fix_object(obj: dict) -> dict:
        """Recursively fix an object schema."""
        obj = obj.copy()

        if obj.get("type") == "object":
            obj["additionalProperties"] = False

            # Make all properties required
            if "properties" in obj:
                obj["required"] = list(obj["properties"].keys())

                # Recursively fix nested objects
                for key, value in obj["properties"].items():
                    if isinstance(value, dict):
                        obj["properties"][key] = fix_object(value)

        # Handle arrays
        if obj.get("type") == "array" and "items" in obj:
            if isinstance(obj["items"], dict):
                obj["items"] = fix_object(obj["items"])

        # Handle $defs (Pydantic v2 uses this for nested models)
        if "$defs" in obj:
            for def_name, def_schema in obj["$defs"].items():
                obj["$defs"][def_name] = fix_object(def_schema)

        return obj

    return fix_object(schema)


# =============================================================================
# BASE PROVIDER CLASS
# =============================================================================

class ModelProvider(ABC):
    """
    Base class for model providers.

    All providers must implement:
    - complete(): Raw text completion
    - stream(): Streaming text completion
    - complete_structured(): Structured output completion (NEW in v2)
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        """
        Generate a text completion.

        Returns:
            dict with keys:
            - content: str (the response text)
            - input_tokens: int
            - output_tokens: int
            - model: str
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream a completion, yielding text chunks."""
        pass

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,  # Lower default for structured output
        retries: int = 3
    ) -> tuple[T, dict]:
        """
        Generate a structured completion.

        NEW IN V2: This is the key addition.

        Args:
            messages: Chat messages
            schema: Pydantic model class to validate against
            system: System prompt
            max_tokens: Max tokens in response
            temperature: Lower = more deterministic (good for structured)
            retries: Number of retry attempts on validation failure

        Returns:
            tuple of (validated_model_instance, usage_dict)
            - validated_model_instance: The response as a Pydantic model
            - usage_dict: {"input_tokens": int, "output_tokens": int}
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models from this provider.

        Returns:
            List of model IDs available from the provider's API.
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        return len(text) // 4


# =============================================================================
# ANTHROPIC PROVIDER (Claude)
# =============================================================================

class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude provider.

    STRUCTURED OUTPUT APPROACH:
    --------------------------
    Claude doesn't have a native "json_schema" response format like OpenAI.
    Instead, we use TOOL USE:

    1. We define a "tool" with our schema as its input_schema
    2. We tell Claude to use this tool (tool_choice: "required")
    3. Claude "calls" the tool with structured data
    4. We extract the tool input and validate with Pydantic

    This is actually MORE reliable than OpenAI's approach because Claude
    is explicitly trying to provide valid tool inputs.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.base_url = "https://api.anthropic.com/v1"
        # Cost per 1K tokens (Sonnet pricing)
        self.cost_per_1k_input = 0.003
        self.cost_per_1k_output = 0.015

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        """Standard text completion."""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            if system:
                payload["system"] = system

            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["content"][0]["text"],
                "input_tokens": data["usage"]["input_tokens"],
                "output_tokens": data["usage"]["output_tokens"],
                "model": self.model
            }

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Streaming text completion."""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
                "stream": True
            }
            if system:
                payload["system"] = system

            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
                timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "content_block_delta":
                            yield data["delta"].get("text", "")

    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retries: int = 3
    ) -> tuple[T, dict]:
        """
        Structured completion using tool use.

        HOW IT WORKS:
        1. We create a "tool" called "structured_response" with the schema
        2. We force Claude to use it with tool_choice
        3. Claude returns tool_use with the structured data
        4. We validate with Pydantic
        """
        # Create a tool definition from the schema
        tool = {
            "name": "structured_response",
            "description": f"Provide a structured response matching the {schema.__name__} schema",
            "input_schema": get_json_schema(schema)
        }

        last_error = None

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    payload = {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": messages,
                        "tools": [tool],
                        "tool_choice": {"type": "tool", "name": "structured_response"}
                    }
                    if system:
                        payload["system"] = system

                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers={
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json=payload,
                        timeout=120.0
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Extract tool use input
                    tool_use = None
                    for block in data["content"]:
                        if block["type"] == "tool_use":
                            tool_use = block
                            break

                    if not tool_use:
                        raise ValueError("No tool_use in response")

                    # Validate with Pydantic
                    validated = schema.model_validate(tool_use["input"])

                    usage = {
                        "input_tokens": data["usage"]["input_tokens"],
                        "output_tokens": data["usage"]["output_tokens"]
                    }

                    return validated, usage

            except ValidationError as e:
                last_error = e
                if attempt < retries - 1:
                    # Add error feedback for retry
                    messages = messages + [
                        {"role": "assistant", "content": f"Tool input: {tool_use['input'] if tool_use else 'none'}"},
                        {"role": "user", "content": f"Validation error: {e}. Please try again with valid data."}
                    ]
            except Exception as e:
                last_error = e
                if attempt == retries - 1:
                    raise

        raise last_error or ValueError("Structured completion failed")

    async def list_models(self) -> list[str]:
        """List available models from Anthropic API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data["data"]]


# =============================================================================
# OPENAI PROVIDER (GPT-4)
# =============================================================================

# Fallback models when the requested model fails (400/404 errors)
OPENAI_MODEL_FALLBACKS = {
    "gpt-5.2": "gpt-4o",
    "gpt-5.1": "gpt-4o",
}


class OpenAIProvider(ModelProvider):
    """
    OpenAI GPT provider.

    STRUCTURED OUTPUT APPROACH:
    --------------------------
    OpenAI has native json_schema support in response_format.
    This is the cleanest implementation.

    We just pass:
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "...", "schema": {...}}
        }

    And OpenAI guarantees valid JSON matching the schema.
    """

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.base_url = "https://api.openai.com/v1"
        # Cost per 1K tokens (GPT-4o pricing)
        self.cost_per_1k_input = 0.005
        self.cost_per_1k_output = 0.015

    def _get_token_param(self, max_tokens: int) -> dict:
        """
        Get the correct token limit parameter for the model.
        GPT-5.x models use 'max_completion_tokens', older models use 'max_tokens'.
        """
        if self.model.startswith("gpt-5"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        """Standard text completion."""
        async with httpx.AsyncClient() as client:
            all_messages = []
            if system:
                all_messages.append({"role": "system", "content": system})
            all_messages.extend(messages)

            payload = {
                "model": self.model,
                "messages": all_messages,
                "temperature": temperature,
                **self._get_token_param(max_tokens)
            }

            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["choices"][0]["message"]["content"],
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
                "model": self.model
            }

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Streaming text completion."""
        async with httpx.AsyncClient() as client:
            all_messages = []
            if system:
                all_messages.append({"role": "system", "content": system})
            all_messages.extend(messages)

            payload = {
                "model": self.model,
                "messages": all_messages,
                "temperature": temperature,
                "stream": True,
                **self._get_token_param(max_tokens)
            }

            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content

    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retries: int = 3
    ) -> tuple[T, dict]:
        """
        Structured completion using json_schema response format.

        OpenAI's native structured output support.
        Includes automatic fallback to alternative models on 400/404 errors.
        """
        last_error = None

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    all_messages = []
                    if system:
                        all_messages.append({"role": "system", "content": system})
                    all_messages.extend(messages)

                    # OpenAI's structured output format
                    # OpenAI has strict requirements - must have additionalProperties: false
                    json_schema = make_openai_compatible_schema(get_json_schema(schema))

                    payload = {
                        "model": self.model,
                        "messages": all_messages,
                        "temperature": temperature,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema.__name__,
                                "strict": True,
                                "schema": json_schema
                            }
                        },
                        **self._get_token_param(max_tokens)
                    }

                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=120.0
                    )
                    response.raise_for_status()
                    data = response.json()

                    content = data["choices"][0]["message"]["content"]

                    # Validate with Pydantic
                    validated = schema.model_validate_json(content)

                    usage = {
                        "input_tokens": data["usage"]["prompt_tokens"],
                        "output_tokens": data["usage"]["completion_tokens"]
                    }

                    return validated, usage

            except httpx.HTTPStatusError as e:
                # Handle model not found errors with fallback
                if e.response.status_code in (400, 404) and self.model in OPENAI_MODEL_FALLBACKS:
                    fallback = OPENAI_MODEL_FALLBACKS[self.model]
                    print(f"Model {self.model} failed ({e.response.status_code}), falling back to {fallback}")
                    self.model = fallback
                    return await self.complete_structured(
                        messages=messages,
                        schema=schema,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        retries=retries
                    )
                last_error = e
                if attempt == retries - 1:
                    raise
            except ValidationError as e:
                last_error = e
                if attempt < retries - 1:
                    messages = messages + [
                        {"role": "assistant", "content": content if 'content' in dir() else ""},
                        {"role": "user", "content": f"Validation error: {e}. Please provide valid JSON."}
                    ]
            except Exception as e:
                last_error = e
                if attempt == retries - 1:
                    raise

        raise last_error or ValueError("Structured completion failed")

    async def list_models(self) -> list[str]:
        """List available models from OpenAI API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data["data"]]


# =============================================================================
# OLLAMA PROVIDER (DeepSeek / Local Models)
# =============================================================================

class OllamaProvider(ModelProvider):
    """
    Ollama local model provider (DeepSeek, Llama, etc.).

    STRUCTURED OUTPUT APPROACH:
    --------------------------
    Ollama doesn't have native structured output support like Claude/OpenAI.
    We use prompt engineering + JSON mode:

    1. Add the schema to the prompt
    2. Request JSON-only output
    3. Parse response, retry on failure

    This is LESS reliable than Claude/OpenAI - expect more retries.
    For critical structured outputs, prefer Claude or OpenAI.
    """

    def __init__(self, model: str = "deepseek-coder-v2:16b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.cost_per_1k_input = 0.0  # Local = free
        self.cost_per_1k_output = 0.0

        # Auto-start Ollama (preserved from v1)
        ensure_ollama_running(model=model, base_url=base_url)

    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        """Standard text completion."""
        async with httpx.AsyncClient() as client:
            # Convert to Ollama's prompt format
            prompt = ""
            if system:
                prompt = f"System: {system}\n\n"
            for msg in messages:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n\n"
            prompt += "Assistant: "

            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    },
                    "stream": False
                },
                timeout=300.0  # Longer timeout for local models
            )
            response.raise_for_status()
            data = response.json()

            return {
                "content": data["response"],
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "model": self.model
            }

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Streaming text completion."""
        async with httpx.AsyncClient() as client:
            prompt = ""
            if system:
                prompt = f"System: {system}\n\n"
            for msg in messages:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n\n"
            prompt += "Assistant: "

            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    },
                    "stream": True
                },
                timeout=300.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retries: int = 3
    ) -> tuple[T, dict]:
        """
        Structured completion using prompt engineering.

        Less reliable than Claude/OpenAI - uses more retries.
        """
        # Build schema instruction
        json_schema = get_json_schema(schema)
        schema_instruction = f"""
You must respond with ONLY valid JSON matching this exact schema:

{json.dumps(json_schema, indent=2)}

IMPORTANT:
- Return ONLY the JSON object, no other text
- No markdown code blocks
- No explanations before or after
- Ensure all required fields are present
"""

        # Combine with system prompt
        full_system = schema_instruction
        if system:
            full_system = f"{system}\n\n{schema_instruction}"

        last_error = None
        last_content = ""

        for attempt in range(retries):
            try:
                result = await self.complete(
                    messages=messages,
                    system=full_system,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                last_content = result["content"]

                # Try to extract JSON
                json_str = extract_json_from_text(last_content)

                # Validate with Pydantic
                validated = schema.model_validate_json(json_str)

                usage = {
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"]
                }

                return validated, usage

            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < retries - 1:
                    # Add error feedback for retry
                    messages = messages + [
                        {"role": "assistant", "content": last_content},
                        {"role": "user", "content": f"Invalid response. Error: {e}\n\nPlease respond with ONLY valid JSON matching the schema. No other text."}
                    ]
            except Exception as e:
                last_error = e
                if attempt == retries - 1:
                    raise

        raise last_error or ValueError("Structured completion failed")

    async def list_models(self) -> list[str]:
        """List available models from Ollama."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/tags",
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_provider(config: dict, provider_name: str) -> ModelProvider:
    """
    Get a provider instance from config.

    This is how the rest of the system gets providers without
    knowing the implementation details.

    Example config:
        models:
          claude:
            provider: "anthropic"
            model: "claude-sonnet-4-20250514"
          deepseek:
            provider: "ollama"
            model: "deepseek-coder-v2:16b"
    """
    provider_config = config.get("models", {}).get(provider_name, {})

    # Better error when model not found in config
    if not provider_config:
        available = list(config.get("models", {}).keys())
        raise ValueError(
            f"Model '{provider_name}' not found in config. "
            f"Available models: {', '.join(available) or 'none'}"
        )

    provider_type = provider_config.get("provider", "")

    # Better error for missing/invalid provider type
    if provider_type not in ("anthropic", "openai", "ollama"):
        raise ValueError(
            f"Model '{provider_name}' has invalid provider type: '{provider_type}'. "
            f"Must be one of: anthropic, openai, ollama"
        )

    if provider_type == "anthropic":
        api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Model '{provider_name}' requires {api_key_env} but it's not set"
            )
        return AnthropicProvider(
            model=provider_config.get("model", "claude-sonnet-4-20250514"),
            api_key=api_key
        )
    elif provider_type == "openai":
        api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Model '{provider_name}' requires {api_key_env} but it's not set"
            )
        return OpenAIProvider(
            model=provider_config.get("model", "gpt-5.2"),
            api_key=api_key
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model=provider_config.get("model", "deepseek-coder-v2:16b"),
            base_url=provider_config.get("base_url", "http://localhost:11434")
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_anthropic(model: str = "claude-sonnet-4-20250514") -> AnthropicProvider:
    """Quick way to get an Anthropic provider."""
    return AnthropicProvider(model=model)


def get_openai(model: str = "gpt-5.2") -> OpenAIProvider:
    """Quick way to get an OpenAI provider."""
    return OpenAIProvider(model=model)


def get_ollama(model: str = "deepseek-coder-v2:16b") -> OllamaProvider:
    """Quick way to get an Ollama provider."""
    return OllamaProvider(model=model)


async def discover_models(config: dict) -> dict[str, list[str]]:
    """
    Discover available models from all configured providers.

    Args:
        config: Configuration dict with models section

    Returns:
        Dict mapping provider type to list of available model IDs.
        Example: {"anthropic": ["claude-..."], "openai": ["gpt-5.2", ...]}
    """
    results = {}
    for name, provider_config in config.get("models", {}).items():
        provider = get_provider(config, name)
        provider_type = provider_config.get("provider")
        try:
            models = await provider.list_models()
            results[provider_type] = models
        except Exception as e:
            results[provider_type] = [f"Error: {e}"]
    return results
