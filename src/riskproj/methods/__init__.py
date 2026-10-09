"""Methods: B0-B6 (baselines and the proposed method) plus the controls C1-C3.

Importing this package registers every method id in
:mod:`riskproj.utils.registry` under the ``method`` kind, so a config naming an
unimplemented arm fails at construction time with a list of what does exist rather
than halfway through a run.

The vocabulary (``08`` §2) is ``train_mode`` (T) crossed with ``inference_mode`` (I):

===========  ==========================================================
B0           frozen Stage-A policy, nothing trained
B1           standard LoRA SFT                    — the reference
B1-LN        B1 with FACT's Logit-Normal schedule — schedule confound
B2 / B2-L    critic penalty during training, fixed λ / dual ascent
B3           activation projection (train + inference) — the method
B1+TTP       projection at inference only          (A7: filter vs regulariser)
B3-trainonly projection during training only       (A7, other half)
B4a / B4b    action-space projection, post-hoc / in-flow (A6: latent vs action)
B5           critic best-of-N                      (A4: gradient vs selection)
B6           critic gate only, no correction       (A3: is gating doing the work?)
C1/C2/C3     random direction / shrink Δh / permuted critic (A1-A2)
===========  ==========================================================
"""

from __future__ import annotations

from riskproj.methods.base import (
    MethodBase,
    MethodResult,
    PolicyWrapper,
    apply_method_controls,
    base_action_from,
    build_method,
    critic_context_from_batch,
    flow_matching_loss,
    lora_disabled,
    next_batch,
    policy_horizon,
    prepare_flow_inputs,
    run_inference,
)
from riskproj.methods import (  # noqa: F401  (import = registration)
    action_proj,
    best_of_n,
    controls,
    lora_sft,
    penalty,
)
from riskproj.utils.registry import available

#: Every method id the experiments in ``11`` refer to. Kept explicit so that a
#: missing registration is an import-time error, and so the eval layer can iterate
#: arms without importing the classes.
METHOD_IDS: tuple[str, ...] = (
    "B0",
    "B1",
    "B1-LN",
    "B1+TTP",
    "B2",
    "B2-L",
    "B3",
    "B3-trainonly",
    "B4a",
    "B4b",
    "B5",
    "B6",
    "C1",
    "C2",
    "C3",
)


def _verify_registry() -> None:
    registered = set(available("method"))
    missing = sorted(set(METHOD_IDS) - registered)
    extra = sorted(registered - set(METHOD_IDS))
    if missing or extra:
        raise RuntimeError(
            "method registry does not match METHOD_IDS: "
            f"missing={missing}, unexpected={extra}. A method that silently is not "
            "registered would make an arm unfalsifiable."
        )


_verify_registry()

__all__ = [
    "METHOD_IDS",
    "MethodBase",
    "MethodResult",
    "PolicyWrapper",
    "apply_method_controls",
    "available_methods",
    "base_action_from",
    "build_method",
    "critic_context_from_batch",
    "flow_matching_loss",
    "lora_disabled",
    "next_batch",
    "policy_horizon",
    "prepare_flow_inputs",
    "run_inference",
]


def available_methods() -> list[str]:
    """Registered method ids (sorted)."""
    return available("method")
