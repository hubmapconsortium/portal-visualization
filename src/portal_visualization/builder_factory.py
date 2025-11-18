from .assays import MALDI_IMS, NANODESI, SALMON_RNASSEQ_SLIDE, SEQFISH
from .builders.base_builders import NullViewConfBuilder


def _lazy_import_builder(builder_name):
    """Lazy import builder classes to avoid loading heavy dependencies.

    This allows the has_visualization function to work without requiring
    vitessce, zarr, and other heavy dependencies to be installed.

    :param str builder_name: The name of the builder class to import
    :return: The builder class
    :rtype: type

    >>> builder = _lazy_import_builder('NullViewConfBuilder')
    >>> builder.__name__
    'NullViewConfBuilder'
    """
    if builder_name == "NullViewConfBuilder":
        return NullViewConfBuilder

    # Import all builders at once when needed
    from .builders.anndata_builders import (
        MultiomicAnndataZarrViewConfBuilder,
        RNASeqAnnDataZarrViewConfBuilder,
        SpatialMultiomicAnnDataZarrViewConfBuilder,
        SpatialRNASeqAnnDataZarrViewConfBuilder,
        XeniumMultiomicAnnDataZarrViewConfBuilder,
    )
    from .builders.imaging_builders import (
        EpicSegImagePyramidViewConfBuilder,
        GeoMxImagePyramidViewConfBuilder,
        ImagePyramidViewConfBuilder,
        IMSViewConfBuilder,
        KaggleSegImagePyramidViewConfBuilder,
        NanoDESIViewConfBuilder,
        SeqFISHViewConfBuilder,
    )
    from .builders.object_by_analyte_builders import ObjectByAnalyteConfBuilder
    from .builders.scatterplot_builders import (
        ATACSeqViewConfBuilder,
        RNASeqViewConfBuilder,
    )
    from .builders.sprm_builders import (
        MultiImageSPRMAnndataViewConfBuilder,
        StitchedCytokitSPRMViewConfBuilder,
        TiledSPRMViewConfBuilder,
    )

    # Map builder names to classes
    builders = {
        "MultiomicAnndataZarrViewConfBuilder": MultiomicAnndataZarrViewConfBuilder,
        "RNASeqAnnDataZarrViewConfBuilder": RNASeqAnnDataZarrViewConfBuilder,
        "SpatialMultiomicAnnDataZarrViewConfBuilder": SpatialMultiomicAnnDataZarrViewConfBuilder,
        "SpatialRNASeqAnnDataZarrViewConfBuilder": SpatialRNASeqAnnDataZarrViewConfBuilder,
        "XeniumMultiomicAnnDataZarrViewConfBuilder": XeniumMultiomicAnnDataZarrViewConfBuilder,
        "EpicSegImagePyramidViewConfBuilder": EpicSegImagePyramidViewConfBuilder,
        "GeoMxImagePyramidViewConfBuilder": GeoMxImagePyramidViewConfBuilder,
        "ImagePyramidViewConfBuilder": ImagePyramidViewConfBuilder,
        "IMSViewConfBuilder": IMSViewConfBuilder,
        "KaggleSegImagePyramidViewConfBuilder": KaggleSegImagePyramidViewConfBuilder,
        "NanoDESIViewConfBuilder": NanoDESIViewConfBuilder,
        "SeqFISHViewConfBuilder": SeqFISHViewConfBuilder,
        "ObjectByAnalyteConfBuilder": ObjectByAnalyteConfBuilder,
        "ATACSeqViewConfBuilder": ATACSeqViewConfBuilder,
        "RNASeqViewConfBuilder": RNASeqViewConfBuilder,
        "MultiImageSPRMAnndataViewConfBuilder": MultiImageSPRMAnndataViewConfBuilder,
        "StitchedCytokitSPRMViewConfBuilder": StitchedCytokitSPRMViewConfBuilder,
        "TiledSPRMViewConfBuilder": TiledSPRMViewConfBuilder,
    }

    if builder_name in builders:
        return builders[builder_name]
    else:  # pragma: no cover
        raise ValueError(f"Unknown builder: {builder_name}")


# This function processes the hints and returns a tuple of booleans
# indicating which builder to use for the given entity.
def process_hints(hints):
    hints = set(hints)
    is_image = "is_image" in hints
    is_rna = "rna" in hints
    is_atac = "atac" in hints
    is_sprm = "sprm" in hints
    is_codex = "codex" in hints
    is_anndata = "anndata" in hints
    is_json = "json_based" in hints
    is_spatial = "spatial" in hints
    is_support = "is_support" in hints
    is_seg_mask = "segmentation_mask" in hints
    is_geomx = "geomx" in hints
    is_xenium = "xenium" in hints
    is_epic = "epic" in hints

    return (
        is_image,
        is_rna,
        is_atac,
        is_sprm,
        is_codex,
        is_anndata,
        is_json,
        is_spatial,
        is_support,
        is_seg_mask,
        is_geomx,
        is_xenium,
        is_epic,
    )


# This function is the main entrypoint for the builder factory.
# It returns the correct builder for the given entity.
#
# The entity is a dict that contains the entity UUID and metadata.
def get_view_config_builder(entity, get_entity, parent=None, epic_uuid=None):
    """Get the appropriate builder class for an entity.

    Returns a builder class (not an instance) that can be used to generate
    Vitessce configurations for the given entity.

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :param str epic_uuid: EPIC UUID if this is an EPIC-related dataset
    :return: Builder class
    :rtype: type
    """
    builder_name = _get_builder_name(entity, get_entity, parent, epic_uuid)
    return _lazy_import_builder(builder_name)


def _get_builder_name(entity, get_entity, parent=None, epic_uuid=None):
    """Get the name of the appropriate builder for an entity.

    This is the core decision logic that doesn't require importing heavy dependencies.
    Returns the builder class name as a string.

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :param str epic_uuid: EPIC UUID if this is an EPIC-related dataset
    :return: Builder class name
    :rtype: str
    """
    if entity.get("uuid") is None:
        raise ValueError("Provided entity does not have a uuid")
    assay_name = entity.get("soft_assaytype")
    hints = entity.get("vitessce-hints", [])
    (
        is_image,
        is_rna,
        is_atac,
        is_sprm,
        is_codex,
        is_anndata,
        is_json,
        is_spatial,
        is_support,
        is_seg_mask,
        is_geomx,
        is_xenium,
        is_epic,
    ) = process_hints(hints)

    # 'epic" is the only hint for object x analyte EPICs
    if is_epic and len(hints) == 1:
        return "ObjectByAnalyteConfBuilder"

    # vis-lifted image pyramids
    if parent is not None:
        # TODO: For now epic (base image's) support datasets doesn't have any hints
        if is_seg_mask and epic_uuid:
            return "EpicSegImagePyramidViewConfBuilder"
        elif is_seg_mask:
            return "KaggleSegImagePyramidViewConfBuilder"

        elif is_support and is_image:
            ancestor_assaytype = get_entity(parent).get("soft_assaytype")
            if ancestor_assaytype == SEQFISH:
                # e.g. parent  = c6a254b2dc2ed46b002500ade163a7cc
                # e.g. support = 9db61adfc017670a196ea9b3ca1852a0
                return "SeqFISHViewConfBuilder"
            elif ancestor_assaytype == MALDI_IMS:
                # e.g. parent  = 3bc3ad124014a632d558255626bf38c9
                # e.g. support = a6116772446f6d1c1f6b3d2e9735cfe0
                return "IMSViewConfBuilder"
            elif ancestor_assaytype == NANODESI:
                # e.g. parent  = 6b93107731199733f266bbd0f3bc9747
                # e.g. support = e1c4370da5523ab5c9be581d1d76ca20
                return "NanoDESIViewConfBuilder"
            else:
                # e.g. parent  = 8adc3c31ca84ec4b958ed20a7c4f4919
                # e.g. support = f9ae931b8b49252f150d7f8bf1d2d13f
                return "ImagePyramidViewConfBuilder"
        else:
            return "NullViewConfBuilder"

    if is_image:
        if is_rna:
            # e.g. Visium (no probes) [Salmon + Scanpy]
            # sample entity (on dev): 72ec02cf1390428c1e9dc2c88928f5f5
            return "SpatialMultiomicAnnDataZarrViewConfBuilder"
        if is_sprm and is_anndata:
            # e.g. CellDIVE [DeepCell + SPRM]
            # sample entity: c3be5650e93907b68ddbdb22b948db32
            return "MultiImageSPRMAnndataViewConfBuilder"
        if is_codex:
            if is_json:
                # legacy JSON-based dataset, e.g. b69d1e2ad1bf1455eee991fce301b191
                return "TiledSPRMViewConfBuilder"
            # e.g. CODEX [Cytokit + SPRM]
            # sample entity: 43213991a54ce196d406707ffe2e86bd
            return "StitchedCytokitSPRMViewConfBuilder"
        if is_geomx:
            return "GeoMxImagePyramidViewConfBuilder"
        if is_xenium:
            return "XeniumMultiomicAnnDataZarrViewConfBuilder"
    if is_rna:
        # multiomic mudata, e.g. 10x Multiome, SNARE-Seq, etc.
        # e.g. 272789a950b2b5d4b9387a1cf66ad487 on dev
        if is_atac:
            return "MultiomicAnndataZarrViewConfBuilder"
        if is_json:
            # e.g. c019a1cd35aab4d2b4a6ff221e92aaab
            return "RNASeqViewConfBuilder"
        # if not JSON, assume that the entity is AnnData-backed
        # TODO - once "anndata" hint is added to the hints for this assay, use that instead
        if assay_name == SALMON_RNASSEQ_SLIDE:
            # e.g. 2a590db3d7ab1e1512816b165d95cdcf
            return "SpatialRNASeqAnnDataZarrViewConfBuilder"
        # e.g. e65175561b4b17da5352e3837aa0e497
        return "RNASeqAnnDataZarrViewConfBuilder"
    if is_atac:
        # e.g. d4493657cde29702c5ed73932da5317c
        return "ATACSeqViewConfBuilder"

    # any entity with no hints, e.g. 2c2179ea741d3bbb47772172a316a2bf
    return "NullViewConfBuilder"


def has_visualization(entity, get_entity, parent=None, epic_uuid=None):
    """Check if an entity has a visualization without loading heavy dependencies.

    This function works with the thin install (no [full] extras required).

    :param dict entity: Entity response from search index
    :param callable get_entity: Function to retrieve entity by UUID
    :param str parent: Parent entity UUID if this is a support dataset
    :param str epic_uuid: EPIC UUID if this is an EPIC-related dataset
    :return: True if the entity has a visualization, False otherwise
    :rtype: bool
    """
    builder_name = _get_builder_name(entity, get_entity, parent, epic_uuid)
    return builder_name != "NullViewConfBuilder"
