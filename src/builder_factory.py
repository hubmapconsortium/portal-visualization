from .builders.base_builders import NullViewConfBuilder
from .builders.sprm_builders import (
    StitchedCytokitSPRMViewConfBuilder, TiledSPRMViewConfBuilder
)
from .builders.imaging_builders import (
    SeqFISHViewConfBuilder, IMSViewConfBuilder, ImagePyramidViewConfBuilder
)
from .builders.anndata_builders import (
    SpatialRNASeqAnnDataZarrViewConfBuilder, RNASeqAnnDataZarrViewConfBuilder
)
from .builders.scatterplot_builders import (
    RNASeqViewConfBuilder, ATACSeqViewConfBuilder
)
from .assays import (
    SEQFISH,
    MALDI_IMS
)


def get_view_config_builder(entity, assay_map):
    data_types = entity["data_types"]
    assay_objs = [assay_map[dt] for dt in data_types]
    assay_names = [assay.name for assay in assay_objs]
    hints = [hint for assay in assay_objs for hint in assay.vitessce_hints]
    dag_names = [dag['name']
                 for dag in entity['metadata']['dag_provenance_list'] if 'name' in dag]
    if "is_image" in hints:
        if "codex" in hints:
            if ('sprm-to-anndata.cwl' in dag_names):
                return StitchedCytokitSPRMViewConfBuilder
            return TiledSPRMViewConfBuilder
        if SEQFISH in assay_names:
            return SeqFISHViewConfBuilder
        if MALDI_IMS in assay_names:
            return IMSViewConfBuilder
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
