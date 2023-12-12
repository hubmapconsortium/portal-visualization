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
#   "vitessce_hints": [
#     "is_image",
#     "pyramid"
#   ]
# }


def get_ancestor_assaytypes(entity, get_assaytype):
    return [get_assaytype(ancestor.get('uuid')).get('assaytype') for ancestor in entity['immediate_ancestors']]


def get_view_config_builder(entity, get_assaytype):
    assay = get_assaytype(entity.get('uuid'))
    assay_name = assay.get('assaytype')
    hints = assay.get('vitessce_hints', [])
    dag_provenance_list = entity.get('metadata', {}).get('dag_provenance_list', [])
    dag_names = [dag['name']
                 for dag in dag_provenance_list if 'name' in dag]
    print(entity.get('uuid'), assay_name)
    if "is_image" in hints:
        if 'sprm' in hints and 'anndata' in hints:
            return MultiImageSPRMAnndataViewConfBuilder
        if "codex" in hints:
            if ('sprm-to-anndata.cwl' in dag_names):
                return StitchedCytokitSPRMViewConfBuilder
            return TiledSPRMViewConfBuilder
        # Both SeqFISH and IMS were submitted very early on, before the
        # special image pyramid datasets existed.  Their assay names should be in
        # the `entity["data_types"]` while newer ones, like NanoDESI, are in the parents
        if assay_name == SEQFISH:
            return SeqFISHViewConfBuilder
        if assay_name == MALDI_IMS:
            return IMSViewConfBuilder
        if NANODESI in [assaytype for assaytype in get_ancestor_assaytypes(entity, get_assaytype)]:
            return NanoDESIViewConfBuilder
        return ImagePyramidViewConfBuilder
    if "rna" in hints:
        # This is the zarr-backed anndata pipeline.
        if "anndata-to-ui.cwl" in dag_names:
            if assay_name == SALMON_RNASSEQ_SLIDE:
                return SpatialRNASeqAnnDataZarrViewConfBuilder
            return RNASeqAnnDataZarrViewConfBuilder
        return RNASeqViewConfBuilder
    if "atac" in hints:
        return ATACSeqViewConfBuilder
    return NullViewConfBuilder


def has_visualization(entity, get_assaytype):
    builder = get_view_config_builder(entity, get_assaytype)
    return builder != NullViewConfBuilder
