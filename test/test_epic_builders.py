# import pytest
# from src.portal_visualization.epic_factory import get_epic_builder
# from src.portal_visualization.builders.base_builders import ConfCells

# @pytest.mark.parametrize(
#     "parent_conf, epic_uuid, expected",
#     [
#         ("epic_uuid", "SegmentationMaskBuilder"),
#     ],
# )
# def test_get_epic_builder(parent_conf, epic_uuid, expected):
#     assert get_epic_builder(parent_conf, epic_uuid).__name__ == expected
