# This function will determine which builder to use for the given entity.
# Since we only have one builder for EPICs right now, we can just return it.
def get_epic_builder(epic_uuid):
    """Get the EPIC builder class.

    Requires [full] install as it imports visualization builders.

    :param str epic_uuid: UUID of the EPIC dataset
    :return: Builder class for EPIC segmentation masks
    :rtype: type
    """
    if epic_uuid is None:
        raise ValueError("epic_uuid must be provided")

    # Lazy import to avoid requiring full dependencies at module load time
    from portal_visualization.builders.epic_builders import SegmentationMaskBuilder

    return SegmentationMaskBuilder
