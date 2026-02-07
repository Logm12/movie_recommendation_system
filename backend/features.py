import hashlib
import os
from typing import Union


class FeatureFlags:
    """configuration for feature flags."""

    # A/B Testing: Split users into Control vs Treatment
    ENABLE_AB_TESTING = True

    # GenAI Explanations: Show "Why this?"
    ENABLE_EXPLANATIONS = True

    @staticmethod
    def is_enabled(flag_name: str) -> bool:
        return getattr(FeatureFlags, flag_name, False)


def get_user_experiment_group(user_id: Union[int, str]) -> str:
    """
    Deterministically assign a user to an experiment group.

    Strategy:
    - Use SHA256 of user_id to ensure consistency.
    - Modulo 2 to split 50/50.

    Returns:
        "control" or "treatment"
    """
    if not FeatureFlags.ENABLE_AB_TESTING:
        return "control"

    # Ensure consistent hashing
    hash_input = str(user_id).encode("utf-8")
    hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)

    if hash_val % 2 == 0:
        return "control"
    else:
        return "treatment"
