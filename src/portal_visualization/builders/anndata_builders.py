from functools import cached_property

from vitessce import (
    VitessceConfig,
    AnnDataWrapper,
    Component as cm,
    CoordinationType,
)
import zarr

from .base_builders import ViewConfBuilder
from ..utils import get_conf_cells


class RNASeqAnnDataZarrViewConfBuilder(ViewConfBuilder):
    """Wrapper class for creating a AnnData-backed view configuration
    for "second generation" post-August 2020 RNA-seq data from anndata-to-ui.cwl like
    https://portal.hubmapconsortium.org/browse/dataset/e65175561b4b17da5352e3837aa0e497
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Spatially resolved RNA-seq assays require some special handling,
        # and others do not.
        self._is_spatial = False
        self._scatterplot_w = 6 if self.is_annotated else 9
        self._spatial_w = 0

    @cached_property
    def zarr_store(self):
        zarr_path = 'hubmap_ui/anndata-zarr/secondary_analysis.zarr'
        request_init = self._get_request_init() or {}
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        return zarr.open(adata_url, mode='r', storage_options={'client_kwargs': request_init})

    @cached_property
    def is_annotated(self):
        z = self.zarr_store
        return z['uns/annotation_metadata/is_annotated'][()]

    def get_conf_cells(self, marker=None):
        zarr_path = 'hubmap_ui/anndata-zarr/secondary_analysis.zarr'
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        # Use .zgroup file as proxy for whether or not the zarr store is present.
        if f'{zarr_path}/.zgroup' not in file_paths_found:
            message = f'RNA-seq assay with uuid {self._uuid} has no .zarr store at {zarr_path}'
            raise FileNotFoundError(message)
        vc = VitessceConfig(name=self._uuid)
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        # Some of the keys (like marker_genes_for_heatmap) here are from our pipeline
        # https://github.com/hubmapconsortium/portal-containers/blob/master/containers/anndata-to-ui
        # while others come from Matt's standard scanpy pipeline
        # or AnnData default (like X_umap or X).
        cell_set_obs = []
        cell_set_obs_names = []
        dags = [
            dag for dag in self._entity['metadata']['dag_provenance_list']
            if 'name' in dag]
        z = self.zarr_store
        if (any(['azimuth-annotate' in dag['origin'] for dag in dags])):
            if self.is_annotated:
                if 'predicted.ASCT.celltype' in z['obs']:
                    cell_set_obs.append("predicted.ASCT.celltype")
                    cell_set_obs_names.append("Predicted ASCT Cell Type")
                if 'predicted_label' in z['obs']:
                    cell_set_obs.append("predicted_label")
                    cell_set_obs_names.append("Cell Ontology Annotation")
        cell_set_obs.append("leiden")
        cell_set_obs_names.append("Leiden")
        gene_alias = 'var/hugo_symbol' if 'var' in z and 'hugo_symbol' in z['var'] else None
        dataset = vc.add_dataset(name=self._uuid).add_object(AnnDataWrapper(
            adata_url=adata_url,
            mappings_obsm=["X_umap"],
            mappings_obsm_names=["UMAP"],
            spatial_centroid_obsm=("X_spatial" if self._is_spatial else None),
            cell_set_obs=cell_set_obs,
            cell_set_obs_names=cell_set_obs_names,
            expression_matrix="X",
            matrix_gene_var_filter="marker_genes_for_heatmap",
            factors_obs=[
                "marker_gene_0",
                "marker_gene_1",
                "marker_gene_2",
                "marker_gene_3",
                "marker_gene_4"
            ],
            request_init=self._get_request_init(),
            gene_alias=gene_alias
        ))

        vc = self._setup_anndata_view_config(vc, dataset, marker)
        return get_conf_cells(vc)

    def _setup_anndata_view_config(self, vc, dataset, marker=None):
        scatterplot = vc.add_view(
            cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", x=0, y=0, w=self._scatterplot_w, h=6)
        cell_sets = vc.add_view(
            cm.CELL_SETS,
            dataset=dataset,
            x=self._scatterplot_w + self._spatial_w,
            y=0,
            w=12 - self._scatterplot_w - self._spatial_w,
            h=3
        )
        gene_list = vc.add_view(
            cm.GENES,
            dataset=dataset,
            x=self._scatterplot_w + self._spatial_w,
            y=4,
            w=12 - self._scatterplot_w - self._spatial_w,
            h=3
        )
        cell_sets_expr = vc.add_view(
            cm.CELL_SET_EXPRESSION, dataset=dataset, x=7, y=6, w=5, h=4)
        heatmap = vc.add_view(
            cm.HEATMAP, dataset=dataset, x=0, y=6, w=7, h=4)
        # Adding heatmap to coordination doesn't do anything,
        # but it also doesn't hurt anything.
        # Vitessce feature request to add it:
        # https://github.com/vitessce/vitessce/issues/1298

        self._add_spatial_view(dataset, vc)

        if marker:
            vc.link_views(
                [cell_sets, gene_list, scatterplot, cell_sets_expr, heatmap],
                [CoordinationType.GENE_SELECTION, CoordinationType.CELL_COLOR_ENCODING],
                [[marker], "geneSelection"]
            )

        return vc

    def _add_spatial_view(self, dataset, vc):
        # This class does not have a spatial_view...
        # but the subclass does, and overrides this method.
        pass


class SpatialRNASeqAnnDataZarrViewConfBuilder(RNASeqAnnDataZarrViewConfBuilder):
    """Wrapper class for creating a AnnData-backed view configuration
    for "second generation" post-August 2020 spatial RNA-seq data from anndata-to-ui.cwl like
    https://portal.hubmapconsortium.org/browse/dataset/2a590db3d7ab1e1512816b165d95cdcf
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Spatially resolved RNA-seq assays require some special handling,
        # and others do not.
        self._is_spatial = True
        self._scatterplot_w = 4
        self._spatial_w = 5

    def _add_spatial_view(self, dataset, vc):
        spatial = vc.add_view(
            cm.SPATIAL,
            dataset=dataset,
            x=self._scatterplot_w,
            y=0,
            w=self._spatial_w,
            h=6)
        [cells_layer] = vc.add_coordination('spatialCellsLayer')
        cells_layer.set_value(
            {
                "visible": True,
                "stroked": False,
                "radius": 20,
                "opacity": 1,
            }
        )
        spatial.use_coordination(cells_layer)
