# examples/example_arch_llm_integration.py
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.core.llm.template as lm
import agent_torch.populations.astoria as astoria
from agent_torch.models import covid
import pandas as pd
import torch
import torch.nn as nn

# 1) Define a minimal template for decisions
class DecisionTemplate(lm.Template):
    system_prompt = "Decide an action propensity given agent/job context."
    soc_code = lm.Variable(desc="job id", learnable=False)
    abilities = lm.Variable(desc="abilities required", learnable=True)
    work_context = lm.Variable(desc="work context", learnable=True)

    def __prompt__(self):
        self.prompt_string = (
            "SOC {soc_code}. Abilities {abilities}. Work context {work_context}."
        )

    def __output__(self):
        return "Return a number in [0,1]."

def setup_runner():
    loader = LoadPopulation(astoria)
    sim = Executor(model=covid, pop_loader=loader)
    runner = sim.runner
    runner.init()
    return runner

def main():
    # 2) Build archetype and broadcast to population
    template = DecisionTemplate()
    llm = MockLLM(seed=0)
    arch = Archetype(prompt=template, llm=llm, n_arch=1)

    # Optional: external job dataframe for richer prompts
    try:
        jobs_df = pd.read_pickle("job_data_clean.pkl")
        arch.configure(external_df=jobs_df)
    except Exception:
        pass

    arch.broadcast(population=astoria, match_on="soc_code")

    # 3) Create runner and inject an adapter policy that calls arch.sample()
    runner = setup_runner()

    # Identify the policy entry to override (example path; adjust to your config)
    sub = "0"
    # Prefer 'citizens' agent type if present to match covid transitions
    available_agents = list(runner.config["substeps"][sub]["policy"].keys())
    agent_type = "citizens" if "citizens" in available_agents else available_agents[0]
    policy_name = list(runner.config["substeps"][sub]["policy"][agent_type].keys())[0]

    class LLMPolicyAdapter(nn.Module):
        def __init__(self, archetype: Archetype, action_key: str = "isolation_decision"):
            super().__init__()
            self.arch = archetype
            self.action_key = action_key
        def forward(self, state, observation):
            # Call into LLM archetype for a per-agent decision vector
            vals = self.arch.sample()
            # Ensure shape matches other actions (N, 1)
            if vals.dim() == 1:
                vals = vals.view(-1, 1)
            # Ensure device matches simulation state tensors (avoid CPU/CUDA mix)
            target_device = None
            def _find_device(obj):
                nonlocal target_device
                if target_device is not None:
                    return
                if torch.is_tensor(obj):
                    target_device = obj.device
                    return
                if isinstance(obj, dict):
                    for v in obj.values():
                        _find_device(v)
            _find_device(state)
            if target_device is not None and vals.device != target_device:
                vals = vals.to(target_device)
            # Return per-agent action dict for this agent_type
            return {self.action_key: vals}

    # 4) Swap the policy function entry with a Module adapter
    adapter = LLMPolicyAdapter(arch, action_key="isolation_decision")
    runner.initializer.policy_function[sub][agent_type][policy_name] = adapter

    # 5) Run one episode
    num_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    runner.step(num_steps)

    # Example: inspect last trajectory snapshot
    traj = runner.state_trajectory[-1][-1]
    print("Last environment snapshot keys:", list(traj["environment"].keys()))

if __name__ == "__main__":
    main()