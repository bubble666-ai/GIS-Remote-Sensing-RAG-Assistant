"""
Model providers module for supporting multiple AI models.
Implements interfaces for OpenAI, Anthropic, Google, XAI, and Zhipu AI models.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

import openai
import requests
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Optional imports for additional providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import xai
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False

try:
    import zhipuai
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI model provider."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key or not self.client:
            return False
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "OpenAI",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude model provider."""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
        elif not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not available")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")

        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        if not ANTHROPIC_AVAILABLE:
            return False
        if not self.api_key or not self.client:
            return False
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "Anthropic",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class GoogleProvider(BaseLLMProvider):
    """Google Gemini model provider."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None

        if self.api_key and GOOGLE_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                logger.error(f"Error initializing Google client: {str(e)}")
        elif not GOOGLE_AVAILABLE:
            logger.warning("Google Generative AI package not available")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Google Gemini API."""
        if not self.model:
            raise ValueError("Google client not initialized")

        try:
            # Convert messages to Gemini format
            prompt = ""
            for msg in messages:
                role = msg["role"].upper()
                if role == "SYSTEM":
                    prompt += f"System: {msg['content']}\n\n"
                elif role == "USER":
                    prompt += f"User: {msg['content']}\n\n"
                elif role == "ASSISTANT":
                    prompt += f"Assistant: {msg['content']}\n\n"

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating Google response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Google API is available."""
        if not GOOGLE_AVAILABLE:
            return False
        if not self.api_key or not self.model:
            return False
        try:
            response = self.model.generate_content("test")
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get Google model information."""
        return {
            "provider": "Google",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class XAIProvider(BaseLLMProvider):
    """XAI Grok model provider."""

    def __init__(self, model_name: str = "grok-beta", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("XAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
            except Exception as e:
                logger.error(f"Error initializing XAI client: {str(e)}")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using XAI API."""
        if not self.client:
            raise ValueError("XAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating XAI response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if XAI API is available."""
        if not XAI_AVAILABLE:
            return False
        if not self.api_key or not self.client:
            return False
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get XAI model information."""
        return {
            "provider": "XAI",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class ZhipuProvider(BaseLLMProvider):
    """Zhipu AI GLM model provider."""

    def __init__(self, model_name: str = "glm-4.6", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Zhipu AI API."""
        if not self.api_key:
            raise ValueError("Zhipu API key not provided")

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating Zhipu response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Zhipu API is available."""
        if not ZHIPU_AVAILABLE:
            return False
        if not self.api_key:
            return False
        try:
            messages = [{"role": "user", "content": "test"}]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 5
            }
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get Zhipu model information."""
        return {
            "provider": "Zhipu AI",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class LocalProvider(BaseLLMProvider):
    """Local model provider using HuggingFace transformers."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model_name, temperature, max_tokens)
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load the local model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using local model."""
        if self.model is None:
            self._load_model()

        if not self.model or not self.tokenizer:
            raise ValueError("Local model not available")

        try:
            # Simple text generation (you might want to improve this)
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"

            prompt += "Assistant: "

            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + min(self.max_tokens, 200),
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Error generating local model response: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if local model is available."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "provider": "Local",
            "model": self.model_name,
            "available": self.is_available(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class ModelProviderFactory:
    """Factory for creating model providers."""

    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "xai": XAIProvider,
        "zhipu": ZhipuProvider,
        "local": LocalProvider
    }

    _model_mappings = {
        # OpenAI models
        "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
        "gpt-4": ("openai", "gpt-4"),
        "gpt-4-turbo": ("openai", "gpt-4-turbo-preview"),

        # Anthropic models
        "claude-3-sonnet": ("anthropic", "claude-3-sonnet-20240229"),
        "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
        "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),

        # Google models
        "gemini-2.0-flash": ("google", "gemini-2.0-flash-exp"),
        "gemini-2.5-pro": ("google", "gemini-2.5-pro-preview-03-25"),
        "gemini-pro": ("google", "gemini-pro"),

        # XAI models
        "grok-3": ("xai", "grok-3"),
        "grok-beta": ("xai", "grok-3"),
        "grok-4-fast-reasoning": ("xai", "grok-3"),

        # Zhipu models
        "glm-4.6": ("zhipu", "glm-4.6"),
        "glm-4": ("zhipu", "glm-4"),

        # Local models
        "local": ("local", "microsoft/DialoGPT-medium"),
    }

    @classmethod
    def create_provider(cls, model_name: str, temperature: float = 0.1, max_tokens: int = 1000) -> BaseLLMProvider:
        """Create a model provider based on model name."""
        if model_name not in cls._model_mappings:
            raise ValueError(f"Unknown model: {model_name}")

        provider_name, actual_model = cls._model_mappings[model_name]
        provider_class = cls._providers[provider_name]

        return provider_class(actual_model, temperature, max_tokens)

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        models = {}

        for model_name, (provider_name, actual_model) in cls._model_mappings.items():
            try:
                provider = cls.create_provider(model_name)
                models[model_name] = provider.get_model_info()
            except Exception as e:
                logger.warning(f"Error checking model {model_name}: {str(e)}")
                models[model_name] = {
                    "provider": provider_name,
                    "model": actual_model,
                    "available": False,
                    "error": str(e)
                }

        return models

    @classmethod
    def list_models_by_provider(cls) -> Dict[str, List[str]]:
        """List models grouped by provider."""
        provider_models = {}

        for model_name, (provider_name, _) in cls._model_mappings.items():
            if provider_name not in provider_models:
                provider_models[provider_name] = []
            provider_models[provider_name].append(model_name)

        return provider_models