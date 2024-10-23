import pytest
from src.portal_visualization.epic_factory import get_epic_builder


@pytest.mark.parametrize(
    "epic_uuid, expected",
    [
        ("epic_uuid", "SegmentationMaskBuilder"),
    ],
)
def test_get_epic_builder(epic_uuid, expected):
    assert get_epic_builder(epic_uuid).__name__ == expected
