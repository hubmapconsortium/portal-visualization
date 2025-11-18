import pytest

try:
    from src.portal_visualization.epic_factory import get_epic_builder

    FULL_DEPS_AVAILABLE = True
except ImportError:
    FULL_DEPS_AVAILABLE = False
    # Skip entire module during collection if full dependencies not available
    pytest.skip("requires [full] optional dependencies", allow_module_level=True)

# Mark all tests in this file as requiring [full] dependencies
pytestmark = pytest.mark.requires_full


@pytest.mark.parametrize(
    ("epic_uuid", "expected"),
    [
        ("epic_uuid", "SegmentationMaskBuilder"),
    ],
)
def test_get_epic_builder(epic_uuid, expected):
    assert get_epic_builder(epic_uuid).__name__ == expected


def test_get_epic_builder_no_uuid():
    with pytest.raises(ValueError, match="epic_uuid must be provided"):
        get_epic_builder(None)
