import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional

import anthropic
import openai
from openai import NOT_GIVEN, OpenAI

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, Discussion


def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)


def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)


def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)


class CheatMiniWoBLLM(AbstractChatModel):
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def __init__(self, wait_time=0) -> None:
        self.wait_time = wait_time

    def __call__(self, messages) -> str:
        if self.wait_time > 0:
            print(f"Waiting for {self.wait_time} seconds")
            time.sleep(self.wait_time)

        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return make_assistant_message(answer)


@dataclass
class CheatMiniWoBLLMArgs:
    model_name = "test/cheat_miniwob_click_test"
    max_total_tokens = 10240
    max_input_tokens = 8000
    max_new_tokens = 128
    wait_time: int = 0

    def make_model(self):
        return CheatMiniWoBLLM(self.wait_time)

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenRouterChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class GoogleModelArgs(BaseModelArgs):
    """Serializable object for instantiating a Gemini chat model via Google's
    OpenAI-compatible endpoint."""

    def make_model(self):
        return GoogleChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class DeepSeekModelArgs(BaseModelArgs):
    """Serializable object for instantiating a DeepSeek chat model via the
    official DeepSeek OpenAI-compatible endpoint (``api.deepseek.com``)."""

    def make_model(self):
        return DeepSeekChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class VertexAIGeminiModelArgs(BaseModelArgs):
    """Serializable object for instantiating Gemini via Vertex AI."""

    project_id: Optional[str] = None
    location: Optional[str] = None
    base_url: Optional[str] = None

    def make_model(self):
        if self.model_name.startswith("google/"):
            vertex_model_name = self.model_name
        else:
            vertex_model_name = f"google/{self.model_name}"
        return VertexAIGeminiChatModel(
            model_name=vertex_model_name,
            project_id=self.project_id,
            location=self.location,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = (
        None  # NOTE: deployment_name is deprecated for Azure OpenAI and won't be used.
    )

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


@dataclass
class SelfHostedModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a self-hosted model."""

    model_url: str = None
    token: str = None
    backend: str = "huggingface"
    n_retry_server: int = 4

    def make_model(self):
        if self.backend == "huggingface":
            # currently only huggingface tgi servers are supported
            if self.model_url is None:
                self.model_url = os.environ["AGENTLAB_MODEL_URL"]
            if self.token is None:
                self.token = os.environ["AGENTLAB_MODEL_TOKEN"]
            # Lazy import to avoid importing HF utilities on non-HF paths
            from agentlab.llm.huggingface_utils import HuggingFaceURLChatModel

            return HuggingFaceURLChatModel(
                model_name=self.model_name,
                model_url=self.model_url,
                token=self.token,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                n_retry_server=self.n_retry_server,
                log_probs=self.log_probs,
            )
        elif self.backend == "vllm":
            return VLLMChatModel(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                n_retry_server=self.n_retry_server,
            )
        else:
            raise ValueError(f"Backend {self.backend} is not supported")


@dataclass
class ChatModelArgs(BaseModelArgs):
    """Object added for backward compatibility with the old ChatModelArgs."""

    model_path: str = None
    model_url: str = None
    model_size: str = None
    training_total_tokens: int = None
    hf_hosted: bool = False
    is_model_operational: str = False
    sliding_window: bool = False
    n_retry_server: int = 4
    infer_tokens_length: bool = False
    vision_support: bool = False
    shard_support: bool = True
    extra_tgi_args: dict = None
    tgi_image: str = None
    info: dict = None

    def __post_init__(self):
        import warnings

        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "ChatModelArgs is deprecated and used only for xray. Use one of the specific model args classes instead.",
            DeprecationWarning,
        )
        warnings.simplefilter("default", DeprecationWarning)

    def make_model(self):
        pass


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


class RetryError(Exception):
    pass


def handle_error(error, itr, min_retry_wait_time, max_retry):
    if not isinstance(error, openai.OpenAIError):
        raise error
    logging.warning(
        f"Failed to get a response from the API: \n{error}\n" f"Retrying... ({itr+1}/{max_retry})"
    )
    wait_time = _extract_wait_time(
        error.args[0],
        min_retry_wait_time=min_retry_wait_time,
    )
    logging.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)
    error_type = error.args[0]
    return error_type


class OpenRouterError(openai.OpenAIError):
    pass


class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        pricing_func=None,
        log_probs=False,
        completion_token_param="max_completion_tokens",
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs
        self.completion_token_param = completion_token_param

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        # Get pricing information
        if pricing_func:
            pricings = pricing_func()
            try:
                self.input_cost = float(pricings[model_name]["prompt"])
                self.output_cost = float(pricings[model_name]["completion"])
            except KeyError:
                logging.warning(
                    f"Model {model_name} not found in the pricing information, prices are set to 0. Maybe try upgrading langchain_community."
                )
                self.input_cost = 0.0
                self.output_cost = 0.0
        else:
            self.input_cost = 0.0
            self.output_cost = 0.0

        client_args = client_args or {}
        self.client = client_class(
            api_key=api_key,
            **client_args,
        )

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                # Only forward `logprobs` when explicitly enabled. Some
                # OpenAI-compatible endpoints (e.g. Gemini's) reject the
                # field outright, even when it is set to False.
                create_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    n=n_samples,
                    temperature=temperature,
                )
                create_kwargs[self.completion_token_param] = self.max_tokens
                if self.log_probs:
                    create_kwargs["logprobs"] = self.log_probs
                completion = self.client.chat.completions.create(**create_kwargs)

                if completion.usage is None:
                    raise OpenRouterError(
                        "The completion object does not contain usage information. This is likely a bug in the OpenRouter API."
                    )

                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        if n_samples == 1:
            res = AIMessage(completion.choices[0].message.content)
            if self.log_probs:
                res["log_probs"] = completion.choices[0].log_probs
            return res
        else:
            return [AIMessage(c.message.content) for c in completion.choices]

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
            # "busted_retry_llm": int(not self.success), # not logged if it occurs anyways
        }


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        if max_tokens is None:
            max_tokens = NOT_GIVEN
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            pricing_func=partial(tracking.get_pricing_litellm, model_name=model_name),
            log_probs=log_probs,
        )


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        client_args = {
            "base_url": "https://openrouter.ai/api/v1",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENROUTER_API_KEY",
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openrouter,
            log_probs=log_probs,
        )


class GoogleChatModel(ChatModel):
    """Chat model for Google Gemini via the official OpenAI-compatible endpoint.

    Authenticates with `GEMINI_API_KEY` (preferred) and falls back to
    `GOOGLE_API_KEY` for compatibility with `google-generativeai` setups.
    """

    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        assert api_key, (
            "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in the environment "
            "when using GoogleChatModel."
        )
        client_args = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=partial(tracking.get_pricing_litellm, model_name=model_name),
            log_probs=log_probs,
            completion_token_param="max_tokens",
        )


class DeepSeekChatModel(ChatModel):
    """Chat model for DeepSeek via the official OpenAI-compatible endpoint.

    Authenticates with ``DEEPSEEK_API_KEY``. The endpoint
    ``https://api.deepseek.com/v1`` accepts the same request/response shape
    as the OpenAI Chat Completions API, so we reuse the standard
    ``OpenAI`` client pointed at it.

    Image content blocks (OpenAI-style ``image_url``) are forwarded
    untouched. Whether the upstream model actually consumes them depends
    on the selected DeepSeek model: text-only models will return a 400
    error if a screenshot is included. Use the
    ``benchmarks/deepseek_vision_smoke_test.py`` script to probe what
    the current endpoint accepts.
    """

    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        assert api_key, (
            "DEEPSEEK_API_KEY must be set in the environment when using "
            "DeepSeekChatModel."
        )
        client_args = {
            "base_url": "https://api.deepseek.com/v1",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=partial(tracking.get_pricing_litellm, model_name=model_name),
            log_probs=log_probs,
        )


class VertexAIGeminiChatModel(ChatModel):
    """Chat model for Gemini on Vertex AI via the OpenAI-compatible endpoint.

    Authenticates with Google Cloud Application Default Credentials. Set
    `GOOGLE_APPLICATION_CREDENTIALS` or run `gcloud auth application-default
    login`, then set `VERTEXAI_PROJECT` / `GOOGLE_CLOUD_PROJECT` and optionally
    `VERTEXAI_LOCATION` / `GOOGLE_CLOUD_LOCATION` (defaults to `global`).
    """

    _AUTH_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    def __init__(
        self,
        model_name,
        project_id=None,
        location=None,
        base_url=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        self.project_id = (
            project_id
            or os.getenv("VERTEXAI_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        self.location = (
            location
            or os.getenv("VERTEXAI_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or os.getenv("GOOGLE_CLOUD_REGION")
            or "global"
        )
        self._credentials = None
        self._google_auth_request = None

        api_key = self._get_access_token()
        client_args = {
            "base_url": base_url
            or os.getenv("VERTEXAI_BASE_URL")
            or self._make_openai_base_url(self.project_id, self.location),
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=partial(tracking.get_pricing_litellm, model_name=model_name),
            log_probs=log_probs,
            completion_token_param="max_tokens",
        )

    @classmethod
    def _make_openai_base_url(cls, project_id: str, location: str) -> str:
        assert project_id, (
            "Vertex AI Gemini requires a Google Cloud project. Set "
            "VERTEXAI_PROJECT or GOOGLE_CLOUD_PROJECT, or configure ADC with a "
            "default project."
        )
        if location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/projects/{project_id}/locations/{location}"
            "/endpoints/openapi"
        )

    def _get_access_token(self) -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request as GoogleAuthRequest
        except ImportError as e:
            raise ImportError(
                "google-auth is required for Vertex AI Gemini. Install "
                "`google-auth` or the repo requirements."
            ) from e

        try:
            self._credentials, adc_project_id = google.auth.default(
                scopes=list(self._AUTH_SCOPES)
            )
        except Exception as e:
            raise RuntimeError(
                "Vertex AI Gemini requires Google Cloud Application Default "
                "Credentials. Run `gcloud auth application-default login` or "
                "set GOOGLE_APPLICATION_CREDENTIALS."
            ) from e

        self.project_id = self.project_id or adc_project_id
        self._google_auth_request = GoogleAuthRequest()
        self._refresh_access_token()
        return self._credentials.token

    def _refresh_access_token(self) -> None:
        if self._credentials is None:
            return
        expiry = getattr(self._credentials, "expiry", None)
        refresh_at = None
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            refresh_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        if not self._credentials.valid or self._credentials.expired or (
            refresh_at is not None and expiry <= refresh_at
        ):
            self._credentials.refresh(self._google_auth_request)
            self.api_key = self._credentials.token
            if hasattr(self, "client"):
                self.client.api_key = self.api_key

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        self._refresh_access_token()
        return super().__call__(messages, n_samples=n_samples, temperature=temperature)


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        deployment_name=None,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        assert (
            api_key
        ), "AZURE_OPENAI_API_KEY has to be defined in the environment when using AzureChatModel"
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert (
            endpoint
        ), "AZURE_OPENAI_ENDPOINT has to be defined in the environment when using AzureChatModel"

        if deployment_name is not None:
            logging.info(
                f"Deployment name is deprecated for Azure OpenAI and won't be used. Using model name: {model_name}."
            )

        client_args = {
            "base_url": endpoint,
            "default_query": {"api-version": "preview"},
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openai,
            log_probs=log_probs,
        )


def __getattr__(name: str):
    """Lazy re-export of optional classes to keep imports light.

    This lets users import HuggingFaceURLChatModel from agentlab.llm.chat_api
    without importing heavy dependencies unless actually used.

    Args:
        name: The name of the attribute to retrieve.

    Returns:
        The requested class or raises AttributeError if not found.

    Raises:
        AttributeError: If the requested attribute is not available.
    """
    if name == "HuggingFaceURLChatModel":
        from agentlab.llm.huggingface_utils import HuggingFaceURLChatModel

        return HuggingFaceURLChatModel
    raise AttributeError(name)


class VLLMChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        n_retry_server=4,
        min_retry_wait_time=60,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=n_retry_server,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="VLLM_API_KEY",
            client_class=OpenAI,
            client_args={"base_url": os.getenv("VLLM_API_URL", "http://localhost:8000/v1")},
            pricing_func=None,
        )


class AnthropicChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", str(max_tokens)))
        self.max_retry = max_retry

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        timeout_seconds = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "300"))
        self.timeout_seconds = timeout_seconds
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    @staticmethod
    def _convert_image_url_content(block: dict) -> dict:
        image_url = block["image_url"]
        url = image_url["url"] if isinstance(image_url, dict) else image_url

        if url.startswith("data:image/") and ";base64," in url:
            media_type, data = url.removeprefix("data:").split(";base64,", 1)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    @classmethod
    def _convert_content_to_anthropic(cls, content):
        if isinstance(content, str):
            return content

        converted_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                converted_content.append(cls._convert_image_url_content(block))
            else:
                converted_content.append(block)
        return converted_content

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Convert OpenAI format to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = self._convert_content_to_anthropic(msg["content"])
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": self._convert_content_to_anthropic(msg["content"]),
                    }
                )

        temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retry):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": anthropic_messages,
                    "max_tokens": self.max_tokens,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature

                if system_message:
                    kwargs["system"] = system_message

                # Anthropic's SDK refuses non-streaming requests whose worst-case
                # runtime (estimated from max_tokens) exceeds 10 minutes. Stream
                # only for those large generations; browser actions are short and
                # are more reliable through the normal request path.
                if self.max_tokens <= 8192:
                    response = self.client.messages.create(**kwargs)
                else:
                    response = self._stream_with_watchdog(kwargs)

                # Track usage if available
                if hasattr(tracking.TRACKER, "instance"):
                    tracking.TRACKER.instance(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        0,  # cost calculation would need pricing info
                    )

                return AIMessage(response.content[0].text)

            except Exception as e:
                if attempt == self.max_retry - 1:
                    raise e
                logging.warning(f"Anthropic API error (attempt {attempt + 1}): {e}")
                time.sleep(60)  # Simple retry delay

    def _stream_with_watchdog(self, kwargs):
        """Run Anthropic streaming with a hard wall-clock timeout.

        The SDK timeout has not always interrupted a wedged streaming response,
        and Ray may run the task body outside Python's main thread. Run the
        stream in a daemon thread so the caller can fail/retry on wall-clock
        timeout even when the SDK call itself is stuck in socket I/O.
        """
        if self.timeout_seconds <= 0:
            with self.client.messages.stream(**kwargs) as stream:
                for _ in stream.text_stream:
                    pass
                return stream.get_final_message()

        result_queue = queue.Queue(maxsize=1)

        def _consume_stream():
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    for _ in stream.text_stream:
                        pass
                    result_queue.put((True, stream.get_final_message()))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put((False, exc))

        thread = threading.Thread(target=_consume_stream, daemon=True)
        thread.start()
        try:
            ok, result = result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Anthropic stream exceeded {self.timeout_seconds:.0f}s"
            ) from exc
        if ok:
            return result
        raise result


@dataclass
class AnthropicModelArgs(BaseModelArgs):
    def make_model(self):
        return AnthropicChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


class BedrockChatModel(AnthropicChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        if (
            not os.getenv("AWS_REGION")
            or not os.getenv("AWS_ACCESS_KEY")
            or not os.getenv("AWS_SECRET_KEY")
        ):
            raise ValueError(
                "AWS_REGION, AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment when using BedrockChatModel"
            )

        self.client = anthropic.AnthropicBedrock(
            aws_region=os.getenv("AWS_REGION"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        )


@dataclass
class BedrockModelArgs(BaseModelArgs):
    def make_model(self):
        return BedrockChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
