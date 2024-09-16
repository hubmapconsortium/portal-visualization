from portal_visualization.builders.epic_builders import SegmentationMaskBuilder


# This function will determine which builder to use for the given entity.
# Since we only have one builder for EPICs right now, we can just return it.
def get_epic_builder(epic_uuid):
    return SegmentationMaskBuilder
