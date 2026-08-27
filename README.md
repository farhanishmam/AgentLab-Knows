# AgentLab-Knows

A fork of [AgentLab](https://github.com/ServiceNow/AgentLab) that provides the
`GenericAgent` and `AGENT_*` configs used to run the **Knows** benchmark
(Google Workspace Docs / Sheets / Slides tasks for browser agents).

## Install

**This repo is not standalone for KNOWS runs.** The `knows_*` benchmarks, the
`knows` BrowserGym backend, and the environment/credential setup all live in
[BrowserGym-Knows](https://github.com/farhanishmam/BrowserGym-Knows), which
pulls this repo in as a submodule — **follow its README to set everything up**.
In short:

```bash
git clone https://github.com/farhanishmam/BrowserGym-Knows
cd BrowserGym-Knows
git submodule update --init --recursive   # includes AgentLab-Knows + the benchmark
# then follow BrowserGym-Knows' README: install, then
#   pip uninstall -y browsergym browsergym-core browsergym-experiments browsergym-webarena
#   make install
# (AgentLab pulls upstream browsergym wheels from PyPI, which have no `knows`
#  backend — the uninstall/reinstall makes the editable forks win.)
```

Installing this repo alone (`pip install -e .`) pulls upstream `browsergym`
from PyPI and `Study(benchmark="knows_...")` will fail with an unknown
benchmark.

## Usage

After completing the BrowserGym-Knows setup (credentials in its `.env`,
`./setup.sh` passing):

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
