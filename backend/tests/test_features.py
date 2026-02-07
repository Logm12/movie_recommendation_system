import pytest
from features import FeatureFlags, get_user_experiment_group


def test_feature_flags_default_values():
    """Test that feature flags have correct default values."""
    assert FeatureFlags.ENABLE_AB_TESTING is True
    assert FeatureFlags.ENABLE_EXPLANATIONS is True


def test_experiment_group_hashing():
    """Test that user ID hashing is deterministic and consistent."""
    # User 1 should always be in the same bucket
    group1 = get_user_experiment_group(1)
    group1_again = get_user_experiment_group(1)
    assert group1 == group1_again

    # Check distribution (roughly) or specific known values
    # We expect 'control' or 'treatment'
    assert group1 in ["control", "treatment"]


def test_experiment_group_distribution():
    """Test that groups are somewhat distributed."""
    groups = [get_user_experiment_group(i) for i in range(100)]
    control_count = groups.count("control")
    treatment_count = groups.count("treatment")

    # Should be roughly 50/50, allowing for variance
    assert control_count > 30
    assert treatment_count > 30
