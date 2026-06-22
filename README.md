# AgentLab-Knows

A fork of [AgentLab](https://github.com/ServiceNow/AgentLab) that provides the
`GenericAgent` and `AGENT_*` configs used to run the **Knows** benchmark
(Google Workspace Docs / Sheets / Slides tasks for browser agents).

## Install

```bash
git clone https://github.com/farhanishmam/AgentLab-Knows
cd AgentLab-Knows
pip install -e .
```

## Usage

Use AgentLab's `GenericAgent` with one of the prebuilt `AGENT_*` configs to run
the Knows benchmark:

```python
from agentlab.agents.generic_agent import AGENT_GPT55_AXT  # GenericAgent config
from agentlab.experiments.study import Study

study = Study([AGENT_GPT55_AXT], benchmark="knows_docs_1")
study.run()
```

Pick the `AGENT_*` config that matches your model and observation mode (e.g.
`AGENT_GPT55_AXT`, `AGENT_GEMINI_31_PRO_SCREENSHOT`, `AGENT_DEEPSEEK_V4_PRO_BOTH`).
All configs are defined in
[src/agentlab/agents/generic_agent/agent_configs.py](src/agentlab/agents/generic_agent/agent_configs.py).

See [README_AgentLab.md](README_AgentLab.md) for the full upstream AgentLab docs.
