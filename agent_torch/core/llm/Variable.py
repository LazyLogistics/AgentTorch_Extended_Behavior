"""
User-facing Variable descriptor
===============================

Descriptor to declare variables inside Template subclasses.
- Carries metadata like description, learnable flag, and default value
- Stores resolved values per-instance
- When learnable=True, owns a trainable tensor parameter attached to the
  Template instance; no separate Slot class is needed.
"""

from typing import Any, Optional, Tuple, Callable
import torch
import torch.nn as nn


class Variable:
    def __init__(self, desc: Optional[str] = None, learnable: bool = False, default: Any = None):
        self.desc = desc
        self.learnable = learnable
        self.default = default
        self._name: Optional[str] = None
        # Number of presentation options for P3O (0..num_options-1)
        # 0: skip, 1: direct, 2: labeled, 3: contextual, 4: descriptive
        self.num_options: int = 2

    def __set_name__(self, owner, name: str):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self._name, self.default)

    def __set__(self, instance, value: Any):
        instance.__dict__[self._name] = value

    @property
    def name(self) -> Optional[str]:
        return self._name

    # --- Learnable parameter support (replaces Slot) ---
    def get_parameter(self, instance: Any) -> Optional[nn.Parameter]:
        """Return/create the learnable parameter (logits over options) for this variable on the given instance."""
        if not self.learnable:
            return None
        
        # Auto-discover name if not set
        if self._name is None:
            for name, var in getattr(instance, '_variables', {}).items():
                if var is self:
                    self._name = name
                    break
        
        if self._name is None:
            return None
            
        param_attr = f"__var_param__{self._name}"
        param = getattr(instance, param_attr, None)
        # Debug: Check for parameter sharing
        if self._name in ['skill_1', 'skill_2', 'skill_3'] and hasattr(instance, '__debug_var_params'):
            print(f"DEBUG Variable {self._name}: param_attr={param_attr}, param_id={id(param) if param else 'None'}")
        elif self._name in ['skill_1', 'skill_2', 'skill_3']:
            instance.__debug_var_params = True
        if not isinstance(param, nn.Parameter):
            # Original uses category_index = min(1, num_categories - 1) as default
            init = torch.zeros(self.num_options, dtype=torch.float32)
            if self.num_options > 1:
                init[1] = 1.0  # Bias toward "include" option (index 1)
            param = nn.Parameter(init, requires_grad=True)
            setattr(instance, param_attr, param)
        return param

    def get_p3o_choice(self, mapping: Optional[dict] = None) -> Tuple[int, Callable[[int, dict], str]]:
        """Return P3O placeholder choice tuple: (num_options, lambda(category, data) -> str).

        The lambda implements presentation choices:
          0: skip → ""
          1: direct → value
          2: labeled → "{field_name}: value"
          3: contextual → "with value"
          4: descriptive → "The {field_name} is value"
        For non-learnable variables, the lambda always returns the direct value.
        """

        field_name = self._name

        def map_value(raw: Any) -> str:
            # Map numeric categorical values using mapping.json when available
            if mapping and field_name in mapping and isinstance(raw, (int, float)):
                try:
                    idx = int(raw)
                    choices = mapping[field_name]
                    if 0 <= idx < len(choices):
                        return str(choices[idx])
                except Exception:
                    pass
            # Already string or anything else → stringify
            return "" if raw is None else str(raw)

        def fmt(category: int, data: dict) -> str:
            if not field_name:
                return ""
            raw_value = data.get(field_name)
            if raw_value is None:
                return ""
            value = map_value(raw_value)

            # If not learnable, always direct
            if not self.learnable:
                return value

            if category == 0:
                return ""
            if category == 1:
                # Format skill names with "- Skill: Description" pattern
                if field_name and field_name != 'soc_code':
                    return f"- {value}: {value}"
                return value
            '''
            if category == 2:
                return f"{field_name}: {value}"
            if category == 3:
                return f"with {value}"
            if category == 4:
                return f"The {field_name} is {value}"
            # Fallback
            '''
            return value

        return self.num_options, fmt

    # --- Helpers for P3O optimization over Variable options ---
    def sample_index(self, instance: Any) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample a presentation index using the instance-bound logits.

        Returns (sampled_index, log_prob, entropy).
        For non-learnable variables, returns (1, 0.0, 0.0) which corresponds to direct value.
        """
        if not self.learnable:
            return 1, torch.tensor(0.0), torch.tensor(0.0)
        logits = self.get_parameter(instance)
        if logits is None:
            return 1, torch.tensor(0.0), torch.tensor(0.0)
        # Check for NaN/inf and reset if needed
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: {self._name} has NaN/inf logits, resetting to default")
            logits.data = torch.tensor([0.0, 1.0], dtype=torch.float32)
        
        # Clamp logits to prevent extreme values
        logits.data = torch.clamp(logits.data, min=-10.0, max=10.0)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        return int(idx.item()), dist.log_prob(idx), dist.entropy()

    def get_probabilities(self, instance: Any) -> torch.Tensor:
        """Return softmax probabilities over presentation options for this variable."""
        if not self.learnable:
            probs = torch.zeros(self.num_options, dtype=torch.float32)
            if self.num_options > 1:
                probs[1] = 1.0  # direct value
            return probs
        logits = self.get_parameter(instance)
        return torch.softmax(logits, dim=0) if logits is not None else torch.zeros(self.num_options, dtype=torch.float32)

    def get_best_index(self, instance: Any) -> int:
        """Return argmax option index based on current probabilities."""
        if not self.learnable:
            return 1
        probs = self.get_probabilities(instance)
        return int(torch.argmax(probs).item()) if probs.numel() > 0 else 1

