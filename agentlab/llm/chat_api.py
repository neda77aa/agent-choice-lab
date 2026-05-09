import logging
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import litellm
import openai
from litellm import Timeout, RateLimitError, APIConnectionError, APIError, ServiceUnavailableError, InternalServerError
from huggingface_hub import InferenceClient
from openai import AzureOpenAI, OpenAI

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.huggingface_utils import HFBaseChatModel
from agentlab.llm.llm_utils import AIMessage, Discussion

RETRYABLE_LITELLM_EXCEPTIONS = (
    Timeout,
    RateLimitError,
    APIConnectionError,
    APIError,
    ServiceUnavailableError,
    InternalServerError
)

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
class LiteLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with LiteLLM."""

    def make_model(self):
        return LiteLLMChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
            pricing_func=tracking.get_pricing_litellm,
            additional_drop_params=self.additional_drop_params,
        )

@dataclass
class VertexLiteLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a LiteLLM model backed by Vertex AI.

    This keeps the existing LiteLLM-based provider flow, but injects Vertex-specific
    kwargs (project + location) into the LiteLLM completion call.

    Expected environment variables:
      - VERTEX_PROJECT (required)
      - VERTEX_LOCATION (optional, defaults to "global")
    """

    vertex_project: str = None
    vertex_location: str = "global"
    # Optional Gemini "thinking budget" (tokens). Gemini 2.5+ requires this in
    # the thinkingConfig payload. Default of 16 matches the prior hardcoded
    # value used for Gemini 3 Pro Preview; Gemini 2.5 Pro requires >=128.
    thinking_budget: int = 16

    def make_model(self):
        vertex_project = self.vertex_project or os.getenv("VERTEX_PROJECT")
        if not vertex_project:
            raise ValueError(
                "VERTEX_PROJECT is required for Vertex AI models. "
                "Set env var VERTEX_PROJECT or pass vertex_project in the config."
            )

        vertex_location = self.vertex_location or os.getenv("VERTEX_LOCATION") or "global"

        # LiteLLM Vertex auth kwargs
        provider_kwargs = {
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
        }

        # Vertex model handling
        # - "Gemini" models are first-party Google models on Vertex (NOT prefixed with `vertex_ai/`)
        # - "Partner" MaaS models (Claude, DeepSeek, Qwen, etc.) are routed by LiteLLM using the
        #   `vertex_ai/...` prefix, but some (notably MaaS "-maas" models) are more reliable via
        #   Vertex's OpenAI-compatible endpoint.
        #
        # Special cases:
        # 1) For Vertex location="global", the hostname is `aiplatform.googleapis.com` (no region prefix)
        # 2) Gemini 3 Pro Preview can return only "thoughts" unless a thinking budget is provided.
        # Auto-prefix known Vertex partner models so LiteLLM routes via Vertex
        _VERTEX_PARTNER_PREFIXES = ("claude-", "deepseek-", "qwen-")
        model_name_for_routing = self.model_name
        if not self.model_name.startswith("vertex_ai/") and self.model_name.startswith(_VERTEX_PARTNER_PREFIXES):
            model_name_for_routing = f"vertex_ai/{self.model_name}"
        is_vertex_partner_model = model_name_for_routing.startswith("vertex_ai/")

        # If this is a Vertex MaaS model (DeepSeek/Qwen/etc.), prefer the OpenAI-compatible endpoint.
        # This avoids a class of 404s / routing issues in provider-specific adapters.
        # Example configured model_name: "vertex_ai/deepseek-ai/deepseek-v3.2-maas"
        if is_vertex_partner_model and "maas" in model_name_for_routing:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)

            host = "aiplatform.googleapis.com" if vertex_location == "global" else f"{vertex_location}-aiplatform.googleapis.com"
            provider_kwargs.update(
                {
                    # Force LiteLLM to use OpenAI-compatible codepath
                    "custom_llm_provider": "openai",
                    "api_base": f"https://{host}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi",
                    "api_key": creds.token,
                }
            )
            # Strip prefix to prevent LiteLLM from re-mapping the provider
            model_name = model_name_for_routing.replace("vertex_ai/", "")
        else:
            model_name = model_name_for_routing

            # Gemini-on-Vertex special handling
            if not is_vertex_partner_model:
                if vertex_location == "global":
                    provider_kwargs["api_base"] = (
                        f"https://aiplatform.googleapis.com/v1/projects/{vertex_project}/"
                        f"locations/global/publishers/google/models/{self.model_name}"
                    )
                provider_kwargs["thinkingConfig"] = {
                    "includeThoughts": False,
                    "thinkingBudget": self.thinking_budget,
                }

        return LiteLLMChatModel(
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
            pricing_func=tracking.get_pricing_litellm,
            additional_drop_params=self.additional_drop_params,
            provider_kwargs=provider_kwargs,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = None

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
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

def handle_litellm_error(error, itr, min_retry_wait_time, max_retry):
    if not isinstance(error, RETRYABLE_LITELLM_EXCEPTIONS):
        raise error
    logging.warning(
        f"Failed to get a response from the API: \n{error}\n" f"Retrying... ({itr+1}/{max_retry})"
    )
    wait_time = _extract_wait_time(
        str(error),
        min_retry_wait_time=min_retry_wait_time,
    )
    logging.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)
    error_type = str(error)
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
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs

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
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=n_samples,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    logprobs=self.log_probs,
                )

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


class LiteLLMChatModel(AbstractChatModel):
    # TODO: Deliberately not refactored into ChatModel for now
    def __init__(
        self,
        model_name,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        pricing_func=None,
        log_probs=False,
        additional_drop_params=None,
        provider_kwargs: Optional[dict] = None,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs
        self.additional_drop_params = additional_drop_params or []
        self.provider_kwargs = provider_kwargs or {}

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
                completion = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    logprobs=self.log_probs,
                    additional_drop_params=list(self.additional_drop_params),
                    **(self.provider_kwargs or {}),
                )

                # LiteLLM parses <think></think> and creates a "reasoning_content" field separately.
                # This behavior is undesirable, so we need to reconstruct the original LLM output.
                # https://github.com/BerriAI/litellm/issues/10702
                message = completion.choices[0].message
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    message.content = f"<think>{reasoning}</think>{message.content}"

                if completion.usage is None:
                    raise OpenRouterError(
                        "The completion object does not contain usage information. This is likely a bug in the OpenRouter API."
                    )

                self.success = True
                break
            except Exception as e:
                error_type = handle_litellm_error(e, itr, self.min_retry_wait_time, self.max_retry)
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
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            pricing_func=tracking.get_pricing_openai,
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


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        client_args = {
            "azure_deployment": deployment_name,
            "azure_endpoint": endpoint,
            "api_version": "2024-02-01",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AzureOpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openai,
            log_probs=log_probs,
        )


class HuggingFaceURLChatModel(HFBaseChatModel):
    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        model_url: str,
        token: Optional[str] = None,
        temperature: Optional[int] = 1e-1,
        max_new_tokens: Optional[int] = 512,
        n_retry_server: Optional[int] = 4,
        log_probs: Optional[bool] = False,
    ):
        super().__init__(model_name, base_model_name, n_retry_server, log_probs)
        if temperature < 1e-3:
            logging.warning("Models might behave weirdly when temperature is too low.")
        self.temperature = temperature

        if token is None:
            token = os.environ["TGI_TOKEN"]

        client = InferenceClient(model=model_url, token=token)
        self.llm = partial(client.text_generation, max_new_tokens=max_new_tokens, details=log_probs)


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
            client_args={"base_url": "http://0.0.0.0:8000/v1"},
            pricing_func=None,
        )
