from .builders.base_builders import NullViewConfBuilder
from .builders.sprm_builders import (
    StitchedCytokitSPRMViewConfBuilder, TiledSPRMViewConfBuilder,
    MultiImageSPRMAnndataViewConfBuilder
)
from .builders.imaging_builders import (
    SeqFISHViewConfBuilder,
    IMSViewConfBuilder,
    ImagePyramidViewConfBuilder,
    NanoDESIViewConfBuilder
)
from .builders.anndata_builders import (
    SpatialRNASeqAnnDataZarrViewConfBuilder, RNASeqAnnDataZarrViewConfBuilder
)
from .builders.scatterplot_builders import (
    RNASeqViewConfBuilder, ATACSeqViewConfBuilder
)
from .assays import (
    SEQFISH,
    MALDI_IMS,
    NANODESI,
    SALMON_RNASSEQ_SLIDE
)

# get_assaytype response example:
# {
#   "assaytype": "image_pyramid",
#   "description": "Image Pyramid",
#   "vitessce-hints": [
#     "is_image",
#     "pyramid"
#   ]
# }


def get_ancestor_assaytypes(entity, get_assaytype):
    return [get_assaytype(ancestor).get('assaytype')
            for ancestor
            in entity.get('immediate_ancestors')]


#
# This function is the main entrypoint for the builder factory.
# It returns the correct builder for the given entity.
#
# The entity is a dict that contains the entity UUID and metadata.
# `get_assaytype` is a function which takes an entity UUID and returns
# the assaytype and vitessce-hints for that entity.
def get_view_config_builder(entity, get_assaytype):
    assay = get_assaytype(entity)
    assay_name = assay.get('assaytype')
    hints = assay.get('vitessce-hints', [])
    dag_provenance_list = entity.get('metadata', {}).get('dag_provenance_list', [])
    dag_names = [dag['name']
                 for dag in dag_provenance_list if 'name' in dag]
    if "is_image" in hints:
        if 'sprm' in hints and 'anndata' in hints:
            # e.g. CellDIVE [DeepCell + SPRM]
            # sample entity: c3be5650e93907b68ddbdb22b948db32
            return MultiImageSPRMAnndataViewConfBuilder
        if "codex" in hints:
            if ('sprm-to-anndata.cwl' in dag_names):
                # e.g. 'CODEX [Cytokit + SPRM]
                # sample entity: 43213991a54ce196d406707ffe2e86bd
                return StitchedCytokitSPRMViewConfBuilder
            # Cannot find an example of this in the wild, every CODEX entity has the
            # sprm-to-anndata.cwl in its dag_provenance_list
            return TiledSPRMViewConfBuilder
        # Check types of image pyramid ancestors to determine which builder to use
        ancestor_assaytypes = get_ancestor_assaytypes(entity, get_assaytype)
        # e.g. c6a254b2dc2ed46b002500ade163a7cc
        if SEQFISH in [assaytype for assaytype in ancestor_assaytypes]:
            return SeqFISHViewConfBuilder
        # e.g. 3bc3ad124014a632d558255626bf38c9
        if MALDI_IMS in [assaytype for assaytype in ancestor_assaytypes]:
            return IMSViewConfBuilder
        # e.g. 6b93107731199733f266bbd0f3bc9747
        if NANODESI in [assaytype for assaytype in ancestor_assaytypes]:
            return NanoDESIViewConfBuilder
        # e.g. f9ae931b8b49252f150d7f8bf1d2d13f
        return ImagePyramidViewConfBuilder
    if "rna" in hints:
        # This is the zarr-backed anndata pipeline.
        if "anndata-to-ui.cwl" in dag_names:
            if assay_name == SALMON_RNASSEQ_SLIDE:
                # e.g. 2a590db3d7ab1e1512816b165d95cdcf
                return SpatialRNASeqAnnDataZarrViewConfBuilder
            # e.g. e65175561b4b17da5352e3837aa0e497
            return RNASeqAnnDataZarrViewConfBuilder
        # e.g. c019a1cd35aab4d2b4a6ff221e92aaab
        return RNASeqViewConfBuilder
    if "atac" in hints:
        # e.g. d4493657cde29702c5ed73932da5317c
        return ATACSeqViewConfBuilder
    # any entity with no hints, e.g. 2c2179ea741d3bbb47772172a316a2bf
    return NullViewConfBuilder


def has_visualization(entity, get_assaytype):
    builder = get_view_config_builder(entity, get_assaytype)
    return builder != NullViewConfBuilder
