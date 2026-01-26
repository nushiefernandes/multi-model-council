"""
Model provider integrations for Agent Council.
Supports Anthropic (Claude), OpenAI (Codex/GPT), and Ollama (DeepSeek local).
"""

import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import httpx


def ensure_ollama_running(model: str = None, base_url: str = "http://localhost:11434") -> bool:
    """
    Ensure Ollama is running and optionally pull the model.
    Returns True if Ollama is ready, False otherwise.
    """
    import urllib.request
    import urllib.error

    # Check if Ollama is already running
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        print("✓ Ollama is running")
    except (urllib.error.URLError, ConnectionRefusedError):
        print("Starting Ollama...")
        # Start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait for it to be ready
        for _ in range(30):  # Wait up to 30 seconds
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
            import json
            data = json.loads(resp.read().decode())
            available_models = [m["name"].split(":")[0] for m in data.get("models", [])]
            model_base = model.split(":")[0]

            if model_base not in available_models:
                print(f"Pulling {model}... (this may take a while)")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=False
                )
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


class ModelProvider(ABC):
    """Base class for model providers."""
    
    @abstractmethod
    async def complete(
        self, 
        messages: list[dict], 
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        """Generate a completion."""
        pass
        
    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        pass
        
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models from this provider."""
        pass


class AnthropicProvider(ModelProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.base_url = "https://api.anthropic.com/v1"
        self.cost_per_1k_input = 0.003  # Sonnet pricing
        self.cost_per_1k_output = 0.015
        
    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
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
                        import json
                        data = json.loads(line[6:])
                        if data["type"] == "content_block_delta":
                            yield data["delta"].get("text", "")

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


# Fallback models when the requested model fails (400/404 errors)
OPENAI_MODEL_FALLBACKS = {
    "gpt-5.2": "gpt-4o",
    "gpt-5.1": "gpt-4o",
}


class OpenAIProvider(ModelProvider):
    """OpenAI GPT/Codex provider."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.base_url = "https://api.openai.com/v1"
        self.cost_per_1k_input = 0.005  # GPT-4o pricing
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
        try:
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
        except httpx.HTTPStatusError as e:
            # Handle model not found errors with fallback
            if e.response.status_code in (400, 404) and self.model in OPENAI_MODEL_FALLBACKS:
                fallback = OPENAI_MODEL_FALLBACKS[self.model]
                print(f"Model {self.model} failed ({e.response.status_code}), falling back to {fallback}")
                self.model = fallback
                return await self.complete(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            raise

    async def stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
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
                        import json
                        data = json.loads(line[6:])
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content

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


class OllamaProvider(ModelProvider):
    """Ollama local model provider (for DeepSeek)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.cost_per_1k_input = 0.0  # Local = free
        self.cost_per_1k_output = 0.0

        # Auto-start Ollama and ensure model is available
        ensure_ollama_running(model=model, base_url=base_url)
        
    async def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> dict:
        async with httpx.AsyncClient() as client:
            # Convert to Ollama format
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
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

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


def get_provider(config: dict, provider_name: str) -> ModelProvider:
    """Get a provider instance from config."""
    provider_config = config.get("models", {}).get(provider_name, {})
    provider_type = provider_config.get("provider", "")
    
    if provider_type == "anthropic":
        api_key = os.environ.get(provider_config.get("api_key_env", "ANTHROPIC_API_KEY"))
        return AnthropicProvider(
            model=provider_config.get("model", "claude-sonnet-4-20250514"),
            api_key=api_key
        )
    elif provider_type == "openai":
        api_key = os.environ.get(provider_config.get("api_key_env", "OPENAI_API_KEY"))
        return OpenAIProvider(
            model=provider_config.get("model", "gpt-5.2"),
            api_key=api_key
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model=provider_config.get("model", "deepseek-coder-v2:16b"),
            base_url=provider_config.get("base_url", "http://localhost:11434")
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


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
