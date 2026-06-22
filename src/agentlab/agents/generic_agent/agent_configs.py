"""
Basic flags and agent configurations for generic agents.
"""

import bgym
from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.experiments import args
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .generic_agent import GenericAgentArgs
from .generic_agent_prompt import GenericPromptFlags
from .tmlr_config import BASE_FLAGS, SIGN_IN_INSTRUCTIONS

FLAGS_CUSTOM = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
)


AGENT_CUSTOM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-3.1-8b-instruct"],
    flags=FLAGS_CUSTOM,
)


# GPT-3.5 default config
FLAGS_GPT_3_5 = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,  # too big for most benchmark except miniwob
        use_ax_tree=True,  # very useful
        use_focused_element=True,  # detrimental on minowob according to ablation study
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,  # very detrimental on L1 and miniwob
        use_action_history=True,  # helpful on miniwob
        use_think_history=False,  # detrimental on L1 and miniwob
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,  # doesn't change much
        extract_clickable_tag=False,  # doesn't change much
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,  # usually detrimental
    use_criticise=False,  # usually detrimental
    use_thinking=True,  # very useful
    use_memory=False,
    use_concrete_example=True,  # useful
    use_abstract_example=True,  # useful
    use_hints=True,  # useful
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
)


AGENT_3_5 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-1106"],
    flags=FLAGS_GPT_3_5,
)

# llama3-70b default config
FLAGS_LLAMA3_70B = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
    add_missparsed_messages=True,
)

AGENT_LLAMA3_70B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-3-70b-instruct"],
    flags=FLAGS_LLAMA3_70B,
)
AGENT_LLAMA31_70B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-3.1-70b-instruct"],
    flags=FLAGS_LLAMA3_70B,
)

FLAGS_8B = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=False,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=True,
        ),
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
    add_missparsed_messages=True,
)


AGENT_8B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["meta-llama/Meta-Llama-3-8B-Instruct"],
    flags=FLAGS_8B,
)


AGENT_LLAMA31_8B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-3.1-8b-instruct"],
    flags=FLAGS_8B,
)


# GPT-4o default config
FLAGS_GPT_4o = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
)

AGENT_4o = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o,
)

AGENT_4o_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o,
)

AGENT_AZURE_4o_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o,
)
AGENT_AZURE_4o = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-2024-08-06"],
    flags=FLAGS_GPT_4o,
)
AGENT_AZURE_41 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
)
AGENT_AZURE_41_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-mini-2025-04-14"],
    flags=FLAGS_GPT_4o,
)
AGENT_AZURE_41_NANO = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-nano-2025-04-14"],
    flags=FLAGS_GPT_4o,
)

AGENT_AZURE_5 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-2025-08-07"],
    flags=FLAGS_GPT_4o,
)

AGENT_AZURE_5_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-mini-2025-08-07"],
    flags=FLAGS_GPT_4o,
)

AGENT_AZURE_5_NANO = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-nano-2025-08-07"],
    flags=FLAGS_GPT_4o,
)

AGENT_CLAUDE_SONNET_35 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.5-sonnet:beta"],
    flags=FLAGS_GPT_4o,
)
AGENT_37_SONNET = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.7-sonnet"],
    flags=FLAGS_GPT_4o,
)
# AGENT_o3_MINI = GenericAgentArgs(
#     chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o3-mini-2025-01-31"],
#     flags=FLAGS_GPT_4o,
# )
AGENT_o3_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/openai/o3-mini"],
    flags=FLAGS_GPT_4o,
)

AGENT_o1_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/openai/o1-mini-2024-09-12"],
    flags=FLAGS_GPT_4o,
)
# GPT-4o vision default config
FLAGS_GPT_4o_VISION = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o_VISION.obs.use_screenshot = True
FLAGS_GPT_4o_VISION.obs.use_som = True

AGENT_4o_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_4o_MINI_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_4o_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-2024-08-06"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_4o_MINI_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_41_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_41_MINI_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-mini-2025-04-14"],
    flags=FLAGS_GPT_4o_VISION,
)
AGENT_AZURE_41_NANO_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4.1-nano-2025-04-14"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_5_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-2025-08-07"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_5_MINI_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-mini-2025-08-07"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_AZURE_5_NANO_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-5-nano-2025-08-07"],
    flags=FLAGS_GPT_4o_VISION,
)

AGENT_CLAUDE_SONNET_35_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.5-sonnet:beta"],
    flags=FLAGS_GPT_4o_VISION,
)
AGENT_LLAMA4_17B_INSTRUCT = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-4-maverick"],
    flags=BASE_FLAGS,
)
GPT5_MINI_FLAGS = BASE_FLAGS.copy()
GPT5_MINI_FLAGS.action = dp.ActionFlags(  # action should not be str to work with agentlab-assistant
    action_set=HighLevelActionSetArgs(
        subsets=["bid"],
        multiaction=False,
    )
)

AGENT_GPT5_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-mini-2025-08-07"],
    flags=GPT5_MINI_FLAGS,
)

FLAGS_GPT55 = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=False,
    ),
    use_plan=True,
    use_criticise=False,
    use_thinking=True,
    use_memory=True,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=300_000,
    be_cautious=True,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
)

AGENT_GPT55 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5.5-2026-04-23"],
    flags=FLAGS_GPT55,
)

# Anthropic Claude Opus 4.7 (released Apr 16, 2026). Reuses the same prompt
# flag set as the frontier GPT-5.5 agent so the comparison is apples-to-apples.
AGENT_OPUS_47 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["anthropic/claude-opus-4-7"],
    flags=FLAGS_GPT55,
)

# Google Gemini 3.1 Pro Preview (released Feb 19, 2026), accessed via
# Gemini's API-key based OpenAI-compatible endpoint.
AGENT_GEMINI_31_PRO = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["google/gemini-3.1-pro-preview"],
    flags=FLAGS_GPT55,
)

# DeepSeek V4 Pro accessed via DeepSeek's official OpenAI-compatible
# endpoint (api.deepseek.com). Authenticates with DEEPSEEK_API_KEY.
AGENT_DEEPSEEK_V4_PRO = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["deepseek/deepseek-v4-pro"],
    flags=FLAGS_GPT55,
)

# ---- Frontier-model observation-mode variants ----------------------------
# We keep three observation flavours of `FLAGS_GPT55` so the same prompt
# scaffolding (plan, memory, thinking, 300k token budget) is applied across
# the three observation modes:
#   * AXT only            -> accessibility tree, no screenshot.
#   * Screenshot only     -> screenshot + Set-of-Mark, no axtree.
#   * Both (axtree+image) -> screenshot + Set-of-Mark + axtree.
FLAGS_GPT55_AXT = FLAGS_GPT55  # alias of the AXT-only config above

FLAGS_GPT55_SCREENSHOT = FLAGS_GPT55.copy()
FLAGS_GPT55_SCREENSHOT.obs.use_ax_tree = False
FLAGS_GPT55_SCREENSHOT.obs.use_screenshot = True
FLAGS_GPT55_SCREENSHOT.obs.use_som = True

FLAGS_GPT55_BOTH = FLAGS_GPT55.copy()
FLAGS_GPT55_BOTH.obs.use_ax_tree = True
FLAGS_GPT55_BOTH.obs.use_screenshot = True
FLAGS_GPT55_BOTH.obs.use_som = True


def _make_frontier_agents(model_key: str, name_prefix: str):
    """Build (axt, screenshot, both) GenericAgentArgs for one frontier model.

    The agent_name is suffixed with the obs-mode so result directories from
    the three modes don't clobber each other.
    """
    base_kwargs = dict(chat_model_args=CHAT_MODEL_ARGS_DICT[model_key])

    axt = GenericAgentArgs(flags=FLAGS_GPT55_AXT, **base_kwargs)
    axt.agent_name = f"{name_prefix}_axt"

    screenshot = GenericAgentArgs(flags=FLAGS_GPT55_SCREENSHOT, **base_kwargs)
    screenshot.agent_name = f"{name_prefix}_screenshot"

    both = GenericAgentArgs(flags=FLAGS_GPT55_BOTH, **base_kwargs)
    both.agent_name = f"{name_prefix}_axt_screenshot"

    return axt, screenshot, both


(
    AGENT_GPT55_AXT,
    AGENT_GPT55_SCREENSHOT,
    AGENT_GPT55_BOTH,
) = _make_frontier_agents(
    "openai/gpt-5.5-2026-04-23",
    name_prefix="GenericAgent-gpt-5.5-2026-04-23",
)

(
    AGENT_OPUS_47_AXT,
    AGENT_OPUS_47_SCREENSHOT,
    AGENT_OPUS_47_BOTH,
) = _make_frontier_agents(
    "anthropic/claude-opus-4-7",
    name_prefix="GenericAgent-claude-opus-4-7",
)

(
    AGENT_GEMINI_31_PRO_AXT,
    AGENT_GEMINI_31_PRO_SCREENSHOT,
    AGENT_GEMINI_31_PRO_BOTH,
) = _make_frontier_agents(
    "google/gemini-3.1-pro-preview",
    name_prefix="GenericAgent-gemini-3.1-pro-preview",
)

# DeepSeek V4 Pro is configured as multimodal, so the screenshot and
# axt+screenshot variants preserve image observations and send them through
# the OpenAI-compatible DeepSeek endpoint.
(
    AGENT_DEEPSEEK_V4_PRO_AXT,
    AGENT_DEEPSEEK_V4_PRO_SCREENSHOT,
    AGENT_DEEPSEEK_V4_PRO_BOTH,
) = _make_frontier_agents(
    "deepseek/deepseek-v4-pro",
    name_prefix="GenericAgent-deepseek-v4-pro",
)

DEFAULT_RS_FLAGS = GenericPromptFlags(
    flag_group="default_rs",
    obs=dp.ObsFlags(
        use_html=True,
        use_ax_tree=args.Choice([True, False]),
        use_focused_element=False,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=args.Choice([True, False], p=[0.7, 0.3]),
        use_action_history=True,
        use_think_history=args.Choice([True, False], p=[0.7, 0.3]),
        use_diff=args.Choice([True, False], p=[0.3, 0.7]),
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=args.Choice([True, False]),
        extract_clickable_tag=False,
        extract_coords=args.Choice(["center", "box"]),
        filter_visible_elements_only=args.Choice([True, False], p=[0.3, 0.7]),
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=args.Choice([["bid"], ["bid", "coord"]]),
            multiaction=args.Choice([True, False], p=[0.7, 0.3]),
        ),
        long_description=False,
        individual_examples=False,
    ),
    # drop_ax_tree_first=True, # this flag is no longer active, according to browsergym doc
    use_plan=args.Choice([True, False]),
    use_criticise=args.Choice([True, False], p=[0.7, 0.3]),
    use_thinking=args.Choice([True, False], p=[0.7, 0.3]),
    use_memory=args.Choice([True, False], p=[0.7, 0.3]),
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=args.Choice([True, False], p=[0.7, 0.3]),
    be_cautious=args.Choice([True, False]),
    enable_chat=False,
    max_prompt_tokens=40_000,
    extra_instructions=SIGN_IN_INSTRUCTIONS,
)


RANDOM_SEARCH_AGENT = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=DEFAULT_RS_FLAGS,
)
