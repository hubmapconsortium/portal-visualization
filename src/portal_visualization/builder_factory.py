from .builders.base_builders import NullViewConfBuilder
from .builders.sprm_builders import (
    StitchedCytokitSPRMViewConfBuilder,
    TiledSPRMViewConfBuilder,
    MultiImageSPRMAnndataViewConfBuilder,
)
from .builders.imaging_builders import (
    SeqFISHViewConfBuilder,
    IMSViewConfBuilder,
    ImagePyramidViewConfBuilder,
    NanoDESIViewConfBuilder,
)
from .builders.anndata_builders import (
    MultiomicAnndataZarrViewConfBuilder,
    SpatialRNASeqAnnDataZarrViewConfBuilder,
    RNASeqAnnDataZarrViewConfBuilder,
    SpatialMultiomicAnnDataZarrViewConfBuilder,
)
from .builders.scatterplot_builders import RNASeqViewConfBuilder, ATACSeqViewConfBuilder
from .assays import SEQFISH, MALDI_IMS, NANODESI, SALMON_RNASSEQ_SLIDE


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
    )


# This function is the main entrypoint for the builder factory.
# It returns the correct builder for the given entity.
#
# The entity is a dict that contains the entity UUID and metadata.
# `get_assaytype` is a function which takes an entity UUID and returns
# a dict containing the assaytype and vitessce-hints for that entity.
def get_view_config_builder(entity, get_assaytype, parent=None):
    if entity.get("uuid") is None:
        raise ValueError("Provided entity does not have a uuid")
    assay = get_assaytype(entity)
    assay_name = assay.get("assaytype")
    hints = assay.get("vitessce-hints", [])
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
    ) = process_hints(hints)

    print(entity)
    print(hints)

    # vis-lifted image pyramids
    if parent is not None:
        print('is support? ', is_support)
        print('is image? ', is_image)
        if is_support and is_image:
            print("parent: ", parent)
            ancestor_assaytype = get_assaytype(parent).get("assaytype")
            print("ancestor assaytype: ", ancestor_assaytype)
            if SEQFISH == ancestor_assaytype:
                # e.g. parent  = c6a254b2dc2ed46b002500ade163a7cc
                # e.g. support = 9db61adfc017670a196ea9b3ca1852a0
                return SeqFISHViewConfBuilder
            elif MALDI_IMS == ancestor_assaytype:
                # e.g. parent  = 3bc3ad124014a632d558255626bf38c9
                # e.g. support = a6116772446f6d1c1f6b3d2e9735cfe0
                return IMSViewConfBuilder
            elif NANODESI == ancestor_assaytype:
                # e.g. parent  = 6b93107731199733f266bbd0f3bc9747
                # e.g. support = e1c4370da5523ab5c9be581d1d76ca20
                return NanoDESIViewConfBuilder
            else:
                # e.g. parent  = 8adc3c31ca84ec4b958ed20a7c4f4919
                # e.g. support = f9ae931b8b49252f150d7f8bf1d2d13f
                return ImagePyramidViewConfBuilder
        else:
            return NullViewConfBuilder

    if is_image:
        if is_rna:
            # e.g. Visium (no probes) [Salmon + Scanpy]
            # sample entity (on dev): 72ec02cf1390428c1e9dc2c88928f5f5
            return SpatialMultiomicAnnDataZarrViewConfBuilder
        if is_sprm and is_anndata:
            # e.g. CellDIVE [DeepCell + SPRM]
            # sample entity: c3be5650e93907b68ddbdb22b948db32
            return MultiImageSPRMAnndataViewConfBuilder
        if is_codex:
            if is_json:
                # legacy JSON-based dataset, e.g. b69d1e2ad1bf1455eee991fce301b191
                return TiledSPRMViewConfBuilder
            # e.g. CODEX [Cytokit + SPRM]
            # sample entity: 43213991a54ce196d406707ffe2e86bd
            return StitchedCytokitSPRMViewConfBuilder

    if is_rna:
        # multiomic mudata, e.g. 10x Multiome, SNARE-Seq, etc.
        # e.g. 272789a950b2b5d4b9387a1cf66ad487 on dev
        if is_atac:
            return MultiomicAnndataZarrViewConfBuilder
        if is_json:
            # e.g. c019a1cd35aab4d2b4a6ff221e92aaab
            return RNASeqViewConfBuilder
        # if not JSON, assume that the entity is AnnData-backed
        # TODO - once "anndata" hint is added to the hints for this assay, use that instead
        if assay_name == SALMON_RNASSEQ_SLIDE:
            # e.g. 2a590db3d7ab1e1512816b165d95cdcf
            return SpatialRNASeqAnnDataZarrViewConfBuilder
        # e.g. e65175561b4b17da5352e3837aa0e497
        return RNASeqAnnDataZarrViewConfBuilder
    if is_atac:
        # e.g. d4493657cde29702c5ed73932da5317c
        return ATACSeqViewConfBuilder
    # any entity with no hints, e.g. 2c2179ea741d3bbb47772172a316a2bf
    return NullViewConfBuilder


def has_visualization(entity, get_assaytype, parent=None):
    print("entity: ", entity)
    if (parent is not None):
        print("parent: ", parent)
    builder = get_view_config_builder(entity, get_assaytype, parent)
    return builder != NullViewConfBuilder
