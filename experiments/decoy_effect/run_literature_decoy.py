import argparse
import csv
import datetime as dt
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import litellm
from dotenv import load_dotenv

# Optional: Disable litellm logging if noisy
litellm.suppress_debug_info = True
# Automatically drop unsupported parameters (like logprobs on Claude)
litellm.drop_params = True

def load_stimuli(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def format_options(stimulus: Dict, condition: str, rng: random.Random) -> Tuple[str, str, Dict[str, str], List[Dict]]:
    keys = ["Target", "Rival"]
    if condition == "3_opt":
        keys.append("Decoy")
    
    rng.shuffle(keys)
    labels = ["A", "B", "C"][:len(keys)]
    
    label_to_key = {label: key for label, key in zip(labels, keys)}
    
    lines = []
    images = []
    for label, key in zip(labels, keys):
        attrs = stimulus["options"][key].copy()
        
        image_path = attrs.pop("image", None)
        if image_path:
            images.append({"label": label, "path": image_path})
            
        attr_text = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
        lines.append(f"Option {label}: {attr_text}")
        
    options_text = "\n".join(lines)
    choices_str = "/".join(labels)
    
    return options_text, choices_str, label_to_key, images

def parse_choice(text: str, valid_ids: List[str]) -> str:
    # Look for patterns like "Therefore, I choose A" or "Option B" or just "C"
    # Search backwards for the last single capital letter among valid choices
    text = text.strip()
    if text in valid_ids:
        return text
        
    matches = re.findall(r"\b([A-C])\b", text)
    if matches:
        for m in reversed(matches):
            if m in valid_ids:
                return m
    return ""

import math
import time
import warnings
import imghdr

# Suppress Vertex AI SDK deprecation warnings so they don't flood the terminal
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")


def _extract_litellm_message_text(message) -> str:
    """Best-effort extraction of text from a LiteLLM/OpenAI-style message.

    Some MaaS "thinking" models (e.g., Qwen) may place chain-of-thought in
    `reasoning_content` and sometimes return `content=None` if the generation
    ran out of tokens. We reconstruct a single string so downstream parsing
    doesn't see empty outputs.
    """
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)

    # Normalize empty strings / None
    content = content if content not in (None, "") else None
    reasoning = reasoning if reasoning not in (None, "") else None

    if reasoning and content:
        return f"<think>{reasoning}</think>{content}"
    if content:
        return content
    if reasoning:
        # If there's no final answer, at least return the reasoning so we can
        # see what happened (and potentially parse a choice letter from it).
        return f"<think>{reasoning}</think>"
    return ""

def get_logprobs_for_options(response, valid_labels: List[str], label_to_key: Dict[str, str]) -> Dict[str, float]:
    """
    Extract the log-probabilities of Target, Rival, and Decoy from the first token's top logprobs.
    Returns a dict mapping the option key ('Target', 'Rival', 'Decoy') to its probability.
    """
    probs = {}
    try:
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            # Get the top logprobs for the very first token generated
            first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # OpenAI logprobs list has dicts like {"token": "A", "logprob": -0.1}
            # The token could be 'A', ' A', 'B', etc.
            token_probs = {}
            for item in first_token_logprobs:
                token = item.token.strip()
                if token in valid_labels:
                    # Keep the max logprob for a given letter (e.g. if 'A' and ' A' both appear)
                    if token not in token_probs or item.logprob > token_probs[token]:
                        token_probs[token] = item.logprob
            
            # Convert logprobs to raw probabilities
            for label in valid_labels:
                lp = token_probs.get(label, -99.9) # Very low prob if not in top 20
                probs[f"prob_{label_to_key[label]}"] = math.exp(lp)
    except Exception:
        # Silently pass because some models (like Claude) do not support logprobs
        pass
        
    return probs

import base64

def construct_multimodal_message(prompt: str, images: List[Dict]) -> Dict:
    if not images:
        return {"role": "user", "content": prompt}
    
    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "text", "text": f"Image for Option {img['label']}:"})

        with open(img["path"], "rb") as f:
            raw = f.read()
            base64_image = base64.b64encode(raw).decode("utf-8")

        # IMPORTANT: Determine mime type from the actual bytes.
        # Some files can have a `.jpg` extension while still being PNG-encoded.
        kind = imghdr.what(None, h=raw)
        if kind == "png":
            mime = "image/png"
        elif kind in ("jpeg", "jpg"):
            mime = "image/jpeg"
        else:
            # Fallback: best-effort based on extension
            ext = img["path"].lower().split(".")[-1]
            mime = "image/png" if ext == "png" else "image/jpeg"
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64_image}"
            }
        })
    return {"role": "user", "content": content}

def run_trial(
    model_name: str,
    temperature: float,
    stimulus: Dict,
    condition: str,
    prompting_mode: str,
    rng: random.Random,
    drop_params: List[str],
    provider_kwargs: Dict,
    for_me: bool = False,
    max_tokens_override: int | None = None,
) -> Dict:
    options_text, choices_str, label_to_key, option_images = format_options(stimulus, condition, rng)
    has_images = len(option_images) > 0
    
    messages = []
    
    scenario = stimulus["scenario"]
    
    # Indicate that the phantom decoy is unavailable if applicable
    if condition == "3_opt" and "phantom" in stimulus.get("id", ""):
        decoy_label = next(label for label, key in label_to_key.items() if key == "Decoy")
        scenario += f" (Option {decoy_label} is currently unavailable)."
    
    prompting_mode_label = f"{prompting_mode}_for_me" if for_me else prompting_mode

    # Qwen thinking-variant models produce >10k chars of <think> reasoning by default,
    # which exhausts max_tokens before any visible answer is emitted. Append a brevity
    # instruction to deliberative/knowledge prompts to keep the visible response short.
    # Disclose this Qwen-specific instruction in the writeup.
    is_qwen_thinking_for_prompt = ("qwen" in model_name.lower()) and ("thinking" in model_name.lower() or "qwen-3" in model_name.lower())
    qwen_brevity = " Keep your reasoning brief — at most 3-4 short sentences — then give your final answer." if is_qwen_thinking_for_prompt else ""

    if prompting_mode == "fast":
        if for_me:
            prompt = f"{scenario}\n\nOptions:\n{options_text}\n\nTry choosing one for me. Respond with {choices_str} only. No explanation."
        else:
            prompt = f"{scenario}\n\nOptions:\n{options_text}\n\nWhich do you choose? Respond with {choices_str} only. No explanation."
        messages.append(construct_multimodal_message(prompt, option_images))
        
    elif prompting_mode == "deliberative":
        if for_me:
            prompt = f"{scenario}\n\nOptions:\n{options_text}\n\nTry choosing a product for me. Think step by step about the trade-offs before giving your final answer.{qwen_brevity} Then, finish your response with: 'Therefore, I choose [X]' where [X] is {choices_str}."
        else:
            prompt = f"{scenario}\n\nOptions:\n{options_text}\n\nThink step by step about the trade-offs before giving your final answer.{qwen_brevity} Then, finish your response with: 'Therefore, I choose [X]' where [X] is {choices_str}."
        messages.append(construct_multimodal_message(prompt, option_images))
        
    elif prompting_mode == "knowledge":
        # Phase A: Ask for knowledge
        messages.append({
            "role": "user", 
            "content": "What is the decoy effect? Give an example."
        })
        # Phase A: get a real model-generated explanation. Apply the same provider
        # fixes as the main call: drop_params (e.g. GPT-5 doesn't accept temperature
        # or max_tokens), and bypass litellm via the vertexai SDK for gemini preview
        # models (litellm 404s on those names). Surface the actual exception text
        # so silent fallbacks don't masquerade as real explanations.
        explanation = None
        try:
            if "gemini-3-pro" in model_name:
                import vertexai
                from vertexai.preview.generative_models import GenerativeModel
                v_project = provider_kwargs.get("vertex_project", "avian-album-450120-f6")
                v_location = provider_kwargs.get("vertex_location", "global")
                vertexai.init(project=v_project, location=v_location)
                pure_model_name = model_name.replace("vertex_ai/", "")
                exp_model = GenerativeModel(pure_model_name)
                exp_res = exp_model.generate_content("What is the decoy effect? Give an example.")
                explanation = exp_res.text
            else:
                exp_kwargs = dict(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    # 500 tokens (~2000 chars) gives Gemini room for thinking-budget
                    # overhead plus a complete explanation without silent truncation.
                    max_tokens=500,
                )
                exp_kwargs.update(provider_kwargs)
                for p in drop_params:
                    exp_kwargs.pop(p, None)
                exp_res = litellm.completion(**exp_kwargs)
                explanation = _extract_litellm_message_text(exp_res.choices[0].message)
        except Exception as e:
            print(f"[knowledge phase-A failed] model={model_name} err={type(e).__name__}: {str(e)[:200]}")
            explanation = "The decoy effect (or asymmetric dominance) occurs when people's preference for two options changes when presented with a third option that is asymmetrically dominated."
        if not explanation:
            print(f"[knowledge phase-A returned empty] model={model_name} — falling back to canned explanation")
            explanation = "The decoy effect (or asymmetric dominance) occurs when people's preference for two options changes when presented with a third option that is asymmetrically dominated."
        
        messages.append({"role": "assistant", "content": explanation})
        
        # Phase B: Decision Task
        if for_me:
            prompt = f"Now, consider this scenario:\n{scenario}\n\nOptions:\n{options_text}\n\nTry choosing a product for me. Think step by step about the trade-offs before giving your final answer.{qwen_brevity} Then, finish your response with: 'Therefore, I choose [X]' where [X] is {choices_str}."
        else:
            prompt = f"Now, consider this scenario:\n{scenario}\n\nOptions:\n{options_text}\n\nThink step by step about the trade-offs before giving your final answer.{qwen_brevity} Then, finish your response with: 'Therefore, I choose [X]' where [X] is {choices_str}."
        messages.append(construct_multimodal_message(prompt, option_images))
        
    valid_labels = list(label_to_key.keys())

    # Qwen "thinking" MaaS models often emit long `reasoning_content` before producing a final
    # answer in `content`. If `max_tokens` is too small, we can hit a length stop before a final
    # letter is produced, leaving `content=None`.
    #
    # We treat these models specially:
    # - Allow a larger `max_tokens` even for "fast" mode.
    # - Disable logprobs collection (unnecessary for this model family + can complicate response parsing).
    is_qwen_thinking = ("qwen" in model_name.lower()) and ("thinking" in model_name.lower())

    # Gemini models with a non-zero thinkingBudget consume that budget BEFORE producing
    # visible output. Fast mode's tiny max_tokens (5) leaves nothing for the answer letter,
    # so the response comes back empty. Detect the budget and ensure we leave headroom.
    gemini_thinking_budget = int(
        (provider_kwargs.get("thinkingConfig") or {}).get("thinkingBudget", 0)
    )

    # Make the actual choice call with retry logic for API limits
    max_retries = 5
    base_sleep = 60 # wait 1 minute initially if rate limited
    
    raw_text = ""
    option_probs = {}
    
    for attempt in range(max_retries):
        try:
            if "gemini-3-pro" in model_name:
                # Fallback to direct vertexai sdk for preview models because litellm url parser 404s on custom names
                import vertexai
                from vertexai.preview.generative_models import GenerativeModel
                
                v_project = provider_kwargs.get("vertex_project", "avian-album-450120-f6")
                v_location = provider_kwargs.get("vertex_location", "global")
                
                vertexai.init(project=v_project, location=v_location)
                pure_model_name = model_name.replace("vertex_ai/", "")
                model = GenerativeModel(pure_model_name)
                
                # Convert messages to a single prompt string for vertex SDK
                # For multimodal content, we'd need to extract text parts
                prompt_str_parts = []
                for m in messages:
                    if isinstance(m["content"], str):
                        prompt_str_parts.append(m["content"])
                    else:
                        for part in m["content"]:
                            if part.get("type") == "text":
                                prompt_str_parts.append(part["text"])
                prompt_str = "\n".join(prompt_str_parts)
                
                res = model.generate_content(prompt_str)
                raw_text = res.text
                option_probs = {} # Vertex pure SDK doesn't natively expose logprobs easily yet
            else:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    # Some MaaS "thinking" models (notably Qwen thinking variants) may emit
                    # extra tokens even when instructed to answer with a single letter.
                    # If max_tokens is too small (e.g. 5), we can get an empty `content` and
                    # only `reasoning_content`, which breaks choice parsing.
                    "max_tokens": (
                        max_tokens_override
                        if (max_tokens_override is not None and prompting_mode != "fast")
                        else (
                            2048
                            if (prompting_mode == "fast" and is_qwen_thinking)
                            else (
                                gemini_thinking_budget + 16
                                if (prompting_mode == "fast" and gemini_thinking_budget > 0)
                                else (
                                    64
                                    if (prompting_mode == "fast" and ("thinking" in model_name.lower() or "qwen" in model_name.lower()))
                                    else (5 if prompting_mode == "fast" else 400)
                                )
                            )
                        )
                    ),
                }
                
                # Merge in any Vertex or Google AI provider kwargs!
                kwargs.update(provider_kwargs)
                
                # Force MaaS endpoint if model_name contains maas (like deepseek-v3.2-maas)
                # IMPORTANT: If load_agent_config() already set up an OpenAI-compatible Vertex MaaS endpoint
                # (custom_llm_provider='openai' + api_base + api_key), do NOT override it here.
                if "maas" in model_name and provider_kwargs.get("custom_llm_provider") != "openai":
                    v_project = provider_kwargs.get("vertex_project", "avian-album-450120-f6")
                    v_location = provider_kwargs.get("vertex_location", "global")
                    host = "aiplatform.googleapis.com" if v_location == "global" else f"{v_location}-aiplatform.googleapis.com"
                    kwargs["api_base"] = f"https://{host}/v1beta1/projects/{v_project}/locations/{v_location}/endpoints/openapi"
                    kwargs["custom_llm_provider"] = "vertex_ai"
                
                # Disable logprobs when using images.
                # Many providers/models don't support logprobs for multimodal requests,
                # and we don't need prob_* for visual-only runs.
                if (
                    (not has_images)
                    and (not is_qwen_thinking)
                    and "logprobs" not in drop_params
                    and prompting_mode == "fast"
                ):
                    kwargs["logprobs"] = True
                    # Let the dynamically adjusted provider_kwargs override the default 20 if needed
                    kwargs["top_logprobs"] = provider_kwargs.get("top_logprobs", 20)
                else:
                    # Cleanly strip top_logprobs if we aren't explicitly capturing logprobs to avoid OpenAI crashing
                    kwargs.pop("top_logprobs", None)
                    provider_kwargs.pop("top_logprobs", None)
                
                # Apply drop_params strictly before calling litellm
                for param in drop_params:
                    kwargs.pop(param, None)
                    
                response = litellm.completion(**kwargs)
                raw_text = _extract_litellm_message_text(response.choices[0].message)
                
                # Capture full probabilities for each option directly from logprobs
                option_probs = (
                    get_logprobs_for_options(response, valid_labels, label_to_key)
                    if (prompting_mode == "fast" and (not has_images))
                    else {}
                )
            
            choice_label = parse_choice(raw_text, valid_labels)
            choice_key = label_to_key.get(choice_label, "Invalid")
            
            # If the response doesn't contain a valid choice, treat it as an error to trigger a retry
            if choice_key == "Invalid":
                print(f"Invalid choice parsed from response: '{raw_text}'. Retrying (Attempt {attempt+1}/{max_retries})...")
                continue
            
            # If we get here with a valid choice, it succeeded! Break the retry loop
            break
            
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "limit" in err_str or "too many requests" in err_str:
                sleep_time = base_sleep * (2 ** attempt) # Exponential backoff: 60s, 120s, 240s...
                print(f"Rate limit hit! Sleeping for {sleep_time} seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
                continue # Retry
            else:
                if "logprobs" in err_str and "not allowed" in err_str:
                    print("Model doesn't support logprobs entirely. Dropping logprobs parameter and retrying...")
                    if "logprobs" not in drop_params:
                        drop_params.append("logprobs")
                    continue
                elif "top_logprobs" in err_str and "must be less than or equal to" in err_str:
                    # Dynamically extract the max allowed value or fallback to 5
                    match = re.search(r"less than or equal to (\d+)", err_str)
                    new_val = int(match.group(1)) if match else 5
                    print(f"Model strictly limits top_logprobs. Lowering to {new_val} and retrying...")
                    provider_kwargs["top_logprobs"] = new_val # This will override the default 20 in the loop
                    continue
                    
                print(f"API Error (Non-Retriable): {e}")
                raw_text = ""
                option_probs = {}
                break

    # Final fallback if all retries exhausted
    if "choice_key" not in locals():
        choice_label = parse_choice(raw_text, valid_labels)
        choice_key = label_to_key.get(choice_label, "Invalid")
    
    # Format options for saving in CSV (remove newlines so it's a single line string)
    options_inline = options_text.replace("\n", " | ")
    
    # Capture the exact prompt shown
    full_prompt_parts = []
    for m in messages:
        if isinstance(m["content"], str):
            full_prompt_parts.append(m["content"])
        else:
            for part in m["content"]:
                if part.get("type") == "text":
                    full_prompt_parts.append(part["text"])
    full_prompt = "\n\n".join(full_prompt_parts)
    
    row_data = {
        "stimulus_id": stimulus["id"],
        "condition": condition,
        "prompting_mode": prompting_mode_label,
        "temperature": temperature,
        "option_order": ",".join(label_to_key.values()),
        "options_text": options_inline,
        "full_prompt": full_prompt.replace("\n", "\\n"),
        "raw_text": raw_text.strip().replace("\n", " "),
        "choice_label": choice_label,
        "choice_key": choice_key,
    }
    
    # Append the logprob probabilities for Target, Rival, (Decoy)
    row_data["prob_Target"] = option_probs.get("prob_Target", None)
    row_data["prob_Rival"] = option_probs.get("prob_Rival", None)
    row_data["prob_Decoy"] = option_probs.get("prob_Decoy", None) if condition == "3_opt" else None
    
    return row_data

from omegaconf import OmegaConf

def load_agent_config(agent_name: str) -> Tuple[str, List[str], Dict]:
    """Loads model_name, additional_drop_params, and calculates provider_kwargs manually to avoid torch import crashes."""
    yaml_path = Path(__file__).resolve().parents[2] / "conf" / "agent" / f"{agent_name}.yaml"
    if not yaml_path.exists():
        return agent_name, [], {}
    
    conf = OmegaConf.load(yaml_path)
    if "chat_model_args" not in conf:
        return agent_name, [], {}
        
    chat_args = conf.chat_model_args
    model_name = chat_args.get("model_name", agent_name)
    drop_params = list(chat_args.get("additional_drop_params", []))
    
    provider_kwargs = {}
    
    # If the config specifies VertexLiteLLMModelArgs, emulate its behavior
    if chat_args.get("_target_") == "agentlab.llm.chat_api.VertexLiteLLMModelArgs":
        # Fallback to the project id seen in ADC logs if missing in env
        vertex_project = chat_args.get("vertex_project") or os.getenv("VERTEX_PROJECT") or "avian-album-450120-f6"
            
        vertex_location = chat_args.get("vertex_location") or os.getenv("VERTEX_LOCATION") or "global"
        
        provider_kwargs["vertex_project"] = vertex_project
        provider_kwargs["vertex_location"] = vertex_location
        
        # For Vertex MaaS endpoints (DeepSeek, Qwen), route them through litellm OpenAI proxy bindings
        if "maas" in model_name:
            import google.auth
            import google.auth.transport.requests
            creds, _ = google.auth.default()
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            
            # Extract standard parts to make URL
            location = vertex_location
            project = vertex_project
            
            # Use raw custom_llm_provider to stop litellm from treating it as vertex_ai mapping
            provider_kwargs["custom_llm_provider"] = "openai"
            # Strip vertex_ai/ prefix to avoid litellm intercepting it
            model_name = model_name.replace("vertex_ai/", "")
            
            # Directly pass the OpenAI compatible endpoint matching the bash curl.
            # IMPORTANT: for location="global" the hostname is `aiplatform.googleapis.com` (no region prefix)
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            provider_kwargs["api_base"] = f"https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"
            provider_kwargs["api_key"] = creds.token
            # Note: No thinkingConfig injected for OpenAI proxy calls
        else:
            # Ensure litellm knows it's a vertex model by prepending vertex_ai/ if missing
            if not model_name.startswith("vertex_ai/"):
                model_name = f"vertex_ai/{model_name}"
                
                # Use standard defaults from agentlab but skip for non-Gemini Vertex models because they reject Extra Inputs.
                # thinkingBudget is YAML-overridable: gemini-3-pro accepts 16; gemini-2.5-pro requires 128-32768.
                if "gemini" in model_name.lower():
                    provider_kwargs["thinkingConfig"] = {
                        "includeThoughts": False,
                        "thinkingBudget": int(chat_args.get("thinking_budget", 16)),
                    }

    return model_name, drop_params, provider_kwargs

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a decoy effect literature experiment with an LLM.")
    parser.add_argument("--stimuli", type=Path, default=Path(__file__).parent / "stimuli.json")
    parser.add_argument("--agent", type=str, default="gpt-4o-mini", help="Name of the agent yaml in conf/agent/ (e.g. claude-3-5-sonnet)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per condition/temp/mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "literature_decoy")
    parser.add_argument("--stimulus-id", type=str, default=None, help="Optional: Run only a specific stimulus by ID")
    parser.add_argument("--test-mode", action="store_true", help="Run a small quick test")
    parser.add_argument("--resume-from", type=Path, default=None, help="Path to an existing CSV file to resume from. Will skip already completed samples.")
    parser.add_argument("--for-me", action="store_true", help="Run the study using 'Try choosing a product for me' instead of 'Which do you choose?'")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        choices=["fast", "deliberative", "knowledge"],
                        help="Restrict to a subset of prompting modes (default: all three).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens for deliberative/knowledge modes (default: 400). "
                             "Use ~2500 for verbose models like Claude in knowledge mode.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="Restrict to a subset of temperatures (default: 0.0 0.3 0.7 1.0). "
                             "Useful for models that ignore temperature (e.g. GPT-5).")

    args = parser.parse_args()

    load_dotenv()

    rng = random.Random(args.seed)
    
    model_name, drop_params, provider_kwargs = load_agent_config(args.agent)
    print(f"Loaded agent: {args.agent} -> Model: {model_name} (Dropping: {drop_params})")

    stimuli = load_stimuli(args.stimuli)
    if args.stimulus_id:
        stimuli = [s for s in stimuli if s["id"] == args.stimulus_id]
        if not stimuli:
            print(f"Warning: No stimulus found with id '{args.stimulus_id}'")
            
    conditions = ["2_opt", "3_opt"]
    temperatures = args.temperatures if args.temperatures else [0.0, 0.3, 0.7, 1.0]
    prompting_modes = args.modes if args.modes else ["fast", "deliberative", "knowledge"]
    
    if args.test_mode:
        temperatures = [0.0]
        if not args.modes:
            prompting_modes = ["fast", "deliberative", "knowledge"]
        args.samples = 1
        stimuli = stimuli[:1]

    # Handle resume logic
    completed_counts = {}
    if args.resume_from:
        if not args.resume_from.exists():
            print(f"\n[WARNING] The resume file specified was not found: {args.resume_from}")
            print("Starting a NEW run instead.\n")
            args.out_dir.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = args.out_dir / f"literature_decoy_{args.agent}_{ts}.csv"
            file_mode = "w"
        else:
            print(f"Resuming from {args.resume_from}...")
            valid_rows = []
            with args.resume_from.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    try:
                        # Create a unique key for the condition
                        key = (row["stimulus_id"], row["condition"], float(row["temperature"]), row["prompting_mode"])
                        
                        # Only keep and count rows if the choice_key is valid
                        choice = str(row.get("choice_key", "")).strip()
                        if choice and choice != "Invalid":
                            completed_counts[key] = completed_counts.get(key, 0) + 1
                            valid_rows.append(row)
                    except (KeyError, ValueError):
                        pass
            
            # Rewrite the file with ONLY the valid rows to clean out past failures
            with args.resume_from.open("w", encoding="utf-8", newline="") as f:
                if fieldnames:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(valid_rows)
                    
            csv_path = args.resume_from
            file_mode = "a"
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "literature_decoy_forme" if args.for_me else "literature_decoy"
        csv_path = args.out_dir / f"{prefix}_{args.agent}_{ts}.csv"
        file_mode = "w"
    
    total_runs = len(stimuli) * len(conditions) * len(temperatures) * len(prompting_modes) * args.samples
    print(f"Total API calls planned (including already completed): {total_runs}")
    print(f"Saving output progressively to: {csv_path}")
    
    current = 0
    skipped = 0
    
    # We will open the file in append mode and write row-by-row to prevent data loss on crashes
    with csv_path.open(file_mode, encoding="utf-8", newline="") as f:
        writer = None
        
        for stim in stimuli:
            for cond in conditions:
                for temp in temperatures:
                    for mode in prompting_modes:
                        
                        # Account for the new prompting mode label if we're in 'for_me' mode
                        target_mode = f"{mode}_for_me" if args.for_me else mode
                        
                        # Check how many runs we've already done for this specific configuration
                        key = (stim["id"], cond, float(temp), target_mode)
                        already_done = completed_counts.get(key, 0)
                        
                        runs_to_do = args.samples - already_done
                        if runs_to_do <= 0:
                            skipped += args.samples
                            current += args.samples
                            continue
                            
                        if already_done > 0:
                            skipped += already_done
                            current += already_done
                            
                        for _ in range(runs_to_do):
                            current += 1
                            if current % 50 == 0:
                                print(f"Progress: {current}/{total_runs} (Skipped {skipped})")
                                f.flush() # Force write buffer to disk every 50 calls
                            
                            row = run_trial(model_name, temp, stim, cond, mode, rng, drop_params, provider_kwargs, args.for_me, args.max_tokens)
                            row.update({
                                "model": model_name,
                                "agent": args.agent,
                                "seed": args.seed,
                                "human_target_2_opt": stim["human_target_share_2_opt"],
                                "human_target_3_opt": stim["human_target_share_3_opt"]
                            })
                            
                            # Initialize writer dynamically based on the first row's keys
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                                if file_mode == "w":
                                    writer.writeheader()
                                
                            writer.writerow(row)

    print(f"Finished! Processed {total_runs} runs (Skipped {skipped}) saving to {csv_path}")

if __name__ == "__main__":
    main()
