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
    NANODESI
)


def get_view_config_builder(entity, get_assay):
    data_types = entity["data_types"]
    assay_objs = [get_assay(dt) for dt in data_types]
    assay_names = [assay.name for assay in assay_objs]
    hints = [hint for assay in assay_objs for hint in assay.vitessce_hints]
    dag_provenance_list = entity.get('metadata', {}).get('dag_provenance_list', [])
    dag_names = [dag['name']
                 for dag in dag_provenance_list if 'name' in dag]
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
        if SEQFISH in assay_names:
            return SeqFISHViewConfBuilder
        if MALDI_IMS in assay_names:
            return IMSViewConfBuilder
        if NANODESI in [dt for e in entity["immediate_ancestors"] for dt in e["data_types"]]:
            return NanoDESIViewConfBuilder
        return ImagePyramidViewConfBuilder
    if "rna" in hints:
        # This is the zarr-backed anndata pipeline.
        if "anndata-to-ui.cwl" in dag_names:
            if "salmon_rnaseq_slideseq" in data_types:
                return SpatialRNASeqAnnDataZarrViewConfBuilder
            return RNASeqAnnDataZarrViewConfBuilder
        return RNASeqViewConfBuilder
    if "atac" in hints:
        return ATACSeqViewConfBuilder
    return NullViewConfBuilder


def has_visualization(entity, get_assay):
    builder = get_view_config_builder(entity, get_assay)
    return builder != NullViewConfBuilder
