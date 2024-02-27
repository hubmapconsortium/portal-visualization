from functools import cached_property

from vitessce import (
    VitessceConfig,
    AnnDataWrapper,
    Component as cm,
    CoordinationType as ct,
    ImageOmeTiffWrapper,
    CoordinationLevel as CL
)

import numpy as np
import zarr


from .base_builders import ViewConfBuilder
from ..utils import get_conf_cells

RNA_SEQ_ANNDATA_FACTOR_PATHS = [f"obs/{key}" for key in [
    "marker_gene_0",
    "marker_gene_1",
    "marker_gene_2",
    "marker_gene_3",
    "marker_gene_4"
]]

RNA_SEQ_FACTOR_LABEL_NAMES = [f'Marker Gene {i}' for i in range(len(RNA_SEQ_ANNDATA_FACTOR_PATHS))]


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
        self._obs_set_paths = None
        self._obs_set_names = None
        self._obs_labels_paths = None
        self._obs_labels_names = None
        self._marker = None
        self._gene_alias = None
        self._views = None

    @cached_property
    def zarr_store(self):
        zarr_path = 'hubmap_ui/anndata-zarr/secondary_analysis.zarr'
        request_init = self._get_request_init() or {}
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        return zarr.open(adata_url, mode='r', storage_options={'client_kwargs': request_init})

    @cached_property
    def is_annotated(self):
        z = self.zarr_store
        if 'uns/annotation_metadata/is_annotated' in z:
            return z['uns/annotation_metadata/is_annotated'][()]
        else:
            return False

    def get_conf_cells(self, marker=None):
        zarr_path = 'hubmap_ui/anndata-zarr/secondary_analysis.zarr'
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        # Use .zgroup file as proxy for whether or not the zarr store is present.
        if f'{zarr_path}/.zgroup' not in file_paths_found:
            message = f'RNA-seq assay with uuid {self._uuid} has no .zarr store at {zarr_path}'
            raise FileNotFoundError(message)
        self._set_up_marker_gene(marker)
        self._set_up_obs_labels()
        vc = VitessceConfig(name=self._uuid, schema_version=self._schema_version)
        dataset = self._set_up_dataset(vc)
        vc = self._setup_anndata_view_config(vc, dataset)
        vc = self._link_marker_gene(vc)
        return get_conf_cells(vc)

    def _set_up_marker_gene(self, marker):
        # HUGO symbols are used as the default gene alias, but need to be converted to
        # Ensembl IDs for gene preselection
        z = self.zarr_store
        gene_alias = 'var/hugo_symbol' if 'var' in z and 'hugo_symbol' in z['var'] else None
        if (gene_alias is not None and marker is not None):
            # If user has indicated a marker gene in parameters and we have a hugo_symbol mapping,
            # then we need to convert it to the proper underlying ensembl ID for the dataset
            # in order for the views to reflect the correct gene.

            obs_attrs = z["obs"].attrs.asdict()
            encoding_version = obs_attrs["encoding-version"]

            # Encoding Version 0.1.0
            # https://anndata.readthedocs.io/en/0.7.8/fileformat-prose.html#categorical-arrays
            if (encoding_version == "0.1.0"):
                # Get the list of ensembl IDs from the zarr store
                ensembl_ids_key = z['var'].attrs['_index']
                ensembl_ids = z['var'][ensembl_ids_key]
                # Get the list of hugo symbols
                hugo_symbols = z[gene_alias]
                # Indices in hugo_index_list match to indices of mapped ensembl ID's
                # Values in hugo_index_list match to indices in hugo_categories
                hugo_index_list = hugo_symbols[:]
                # Get the key for the hugo categories list from the categorical entry attributes
                hugo_categories_key = hugo_symbols.attrs.asdict()['categories']
                # Get the list of categories that the hugo_index_list's values map to
                hugo_categories = z['var'][hugo_categories_key][:]
                # Find the index of the user-provided marker gene in the list of hugo symbols
                marker_index_in_categories = np.where(hugo_categories == marker)[0][0]
                # If the user-provided gene's index is found, continue
                if (marker_index_in_categories >= 0):
                    # Find index of HUGO pointer corresponding to marker gene
                    marker_index = np.where(hugo_index_list == marker_index_in_categories)[0][0]
                    # If valid index is found, set the marker name to the corresponding Ensembl ID
                    if (marker_index >= 0):
                        marker = ensembl_ids[marker_index]
                    else:
                        pass
            # Encoding Version 0.2.0
            # https://anndata.readthedocs.io/en/latest/fileformat-prose.html#categorical-arrays
            # Our pipeline currently does not use this encoding version
            # Future improvement to be implemented in CAT-137
            # elif (encoding_version == "0.2.0"):
            #     print('TODO - Encoding Version 0.2.0 support')
        self._marker = marker
        self._gene_alias = gene_alias

    def _set_up_dataset(self, vc):
        adata_url = self._build_assets_url(
            'hubmap_ui/anndata-zarr/secondary_analysis.zarr', use_token=False)
        z = self.zarr_store
        dataset = vc.add_dataset(name=self._uuid).add_object(AnnDataWrapper(
            adata_url=adata_url,
            obs_feature_matrix_path="X",
            initial_feature_filter_path="var/marker_genes_for_heatmap",
            obs_set_paths=self._obs_set_paths,
            obs_set_names=self._obs_set_names,
            obs_locations_path="obsm/X_spatial" if self._is_spatial else None,
            obs_segmentations_path=None,
            obs_embedding_paths=["obsm/X_umap"],
            obs_embedding_names=["UMAP"],
            obs_embedding_dims=[[0, 1]],
            request_init=self._get_request_init(),
            coordination_values=None,
            feature_labels_path='var/hugo_symbol' if 'var' in z else None,
            gene_alias=self._gene_alias,
            obs_labels_paths=self._obs_labels_paths,
            obs_labels_names=self._obs_labels_names
        ))
        return dataset

    def _set_up_obs_labels(self):
        # Some of the keys (like marker_genes_for_heatmap) here are from our pipeline
        # https://github.com/hubmapconsortium/portal-containers/blob/master/containers/anndata-to-ui
        # while others come from Matt's standard scanpy pipeline
        # or AnnData default (like X_umap or X).
        # obs sets are annotated sets of cells/clusters used to color the cells
        obs_set_paths = []
        obs_set_names = []
        # obs labels are tooltip helpers which e.g. identify highly expressed genes
        # or help map predicted cell labels to their IDs
        obs_label_paths = RNA_SEQ_ANNDATA_FACTOR_PATHS
        obs_label_names = RNA_SEQ_FACTOR_LABEL_NAMES
        dags = [
            dag for dag in self._entity['metadata']['dag_provenance_list']
            if 'name' in dag]
        z = self.zarr_store
        if (any(['azimuth-annotate' in dag['origin'] for dag in dags])):
            if self.is_annotated:
                if 'predicted.ASCT.celltype' in z['obs']:
                    obs_set_paths.append("obs/predicted.ASCT.celltype")
                    obs_set_names.append("Predicted ASCT Cell Type")
                if 'predicted_label' in z['obs']:
                    obs_set_paths.append("obs/predicted_label")
                    obs_set_names.append("Cell Ontology Annotation")
                if 'predicted_CLID' in z['obs']:
                    obs_label_paths.append("obs/predicted_CLID")
                    obs_label_names.append("Predicted CL ID")

        obs_set_paths.append("obs/leiden")
        obs_set_names.append("Leiden")
        self._obs_set_paths = obs_set_paths
        self._obs_set_names = obs_set_names
        self._obs_labels_paths = obs_label_paths
        self._obs_labels_names = obs_label_names

    def _setup_anndata_view_config(self, vc, dataset):
        scatterplot = vc.add_view(
            cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", x=0, y=0, w=self._scatterplot_w, h=6)
        cell_sets = vc.add_view(
            cm.OBS_SETS,
            dataset=dataset,
            x=self._scatterplot_w + self._spatial_w,
            y=0,
            w=12 - self._scatterplot_w - self._spatial_w,
            h=3
        )
        gene_list = vc.add_view(
            cm.FEATURE_LIST,
            dataset=dataset,
            x=self._scatterplot_w + self._spatial_w,
            y=4,
            w=12 - self._scatterplot_w - self._spatial_w,
            h=3
        )
        cell_sets_expr = vc.add_view(
            cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset, x=7, y=6, w=5, h=4)
        heatmap = vc.add_view(
            cm.HEATMAP, dataset=dataset, x=0, y=6, w=7, h=4)
        # Adding heatmap to coordination doesn't do anything,
        # but it also doesn't hurt anything.
        # Vitessce feature request to add it:
        # https://github.com/vitessce/vitessce/issues/1298

        # Spatial view is added if present, otherwise gets filtered out before views are linked
        # This ensures that the view config is valid for datasets with and without a spatial view
        spatial = self._add_spatial_view(dataset, vc)

        views = list(filter(lambda v: v is not None, [
                     cell_sets, gene_list, scatterplot, cell_sets_expr, heatmap, spatial]))

        self._views = views

        return vc

    def _add_spatial_view(self, dataset, vc):
        # This class does not have a spatial_view...
        # but the subclass does, and overrides this method.
        return None

    def _link_marker_gene(self, vc):
        # Link top 5 marker genes
        vc.link_views(self._views,
                      [ct.OBS_LABELS_TYPE for _ in self._obs_labels_names],
                      self._obs_labels_names,
                      allow_multiple_scopes_per_type=True)
        # Link user-provided marker gene
        if self._marker:
            vc.link_views(
                self._views,
                [ct.FEATURE_SELECTION, ct.OBS_COLOR_ENCODING],
                [[self._marker], 'geneSelection'],
            )
        return vc


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
        [cells_layer] = vc.add_coordination('spatialSegmentationLayer')
        cells_layer.set_value(
            {
                "visible": True,
                "stroked": False,
                "radius": 20,
                "opacity": 1,
            }
        )
        spatial.use_coordination(cells_layer)
        return spatial


class SpatialMultiomicAnnDataZarrViewConfBuilder(SpatialRNASeqAnnDataZarrViewConfBuilder):
    """
    Wrapper class for creating a AnnData-backed view configuration for multiomic spatial data
    such as Visium.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._scatterplot_w = 3
        self._spatial_w = 3

    def _get_scale_factor(self):
        z = self.zarr_store
        visium_scalefactor_path = 'spatial/visium/scalefactors/spot_diameter_fullres'
        if visium_scalefactor_path in z['uns']:
            return z['uns'][visium_scalefactor_path][()].tolist()

    def _set_up_dataset(self, vc):
        adata_url = self._build_assets_url(
            'hubmap_ui/anndata-zarr/secondary_analysis.zarr', use_token=False)
        image_url = self._build_assets_url(
            'ometiff-pyramids/visium_histology_hires_pyramid.ome.tif', use_token=True)
        # Add dataset with Visium image and secondary analysis anndata
        visium_image = ImageOmeTiffWrapper(
            img_url=image_url,
            uid="visium",
            request_init=self._get_request_init(),
            coordination_values={
                "fileUid": "visium"
            }
        )
        visium_spots = AnnDataWrapper(
            adata_url=adata_url,
            obs_feature_matrix_path="X",
            obs_spots_path="obsm/X_spatial",
            obs_set_paths=self._obs_set_paths,
            obs_set_names=self._obs_set_names,
            obs_labels_names=self._obs_labels_names,
            obs_labels_paths=self._obs_labels_paths,
            feature_labels_path="var/hugo_symbol",
            request_init=self._get_request_init(),
            coordination_values={
                "obsType": "spot",
                "featureType": "gene",
                "featureLabelsType": "gene",
            }
        )
        obs_sets = AnnDataWrapper(
            adata_url=adata_url,
            obs_feature_matrix_path="X",
            obs_set_paths=self._obs_set_paths,
            obs_set_names=self._obs_set_names,
            obs_labels_names=self._obs_labels_names,
            obs_labels_paths=self._obs_labels_paths,
            obs_locations_path="obsm/X_spatial",
            obs_embedding_paths=["obsm/X_umap", "obsm/X_pca"],
            obs_embedding_names=["UMAP", "PCA"],
            obs_embedding_dims=[[0, 1], [0, 1]],
            feature_labels_path="var/hugo_symbol",
            request_init=self._get_request_init(),
            initial_feature_filter_path="var/top_highly_variable",
            coordination_values={
                "obsType": "cell",
                "featureType": "gene",
                "featureLabelsType": "gene",
            }
        )
        dataset = vc.add_dataset(
            name='Visium'
        ).add_object(
            visium_image
        ).add_object(
            visium_spots
        ).add_object(
            obs_sets
        )
        return dataset

    def _setup_anndata_view_config(self, vc, dataset):
        # Add / lay out views
        umap = vc.add_view(
            cm.SCATTERPLOT, dataset=dataset, mapping="UMAP",
            w=3, h=6, x=0, y=0)
        spatial = vc.add_view(
            "spatialBeta", dataset=dataset,
            w=3, h=6, x=3, y=0)
        heatmap = vc.add_view(
            cm.HEATMAP, dataset=dataset,
            w=6, h=6, x=0, y=6
        ).set_props(transpose=True)

        lc = vc.add_view("layerControllerBeta", dataset=dataset,
                         w=6, h=3, x=6, y=0)

        # Add scatterplot, cellsets, gene list, cellsets expression, and heatmap
        cell_sets = vc.add_view(
            cm.OBS_SETS, dataset=dataset,
            w=3, h=4, x=6, y=2)

        gene_list = vc.add_view(
            cm.FEATURE_LIST, dataset=dataset,
            w=3, h=4, x=9, y=2)

        cell_sets_expr = vc.add_view(
            cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset,
            w=6, h=5, x=6, y=7
        )

        all_views = [spatial, lc, umap, cell_sets, cell_sets_expr, gene_list, heatmap]

        self._views = all_views

        spot_views = [spatial, lc]

        # selected_gene_views = [umap, gene_list, heatmap, spatial]

        # Link spatial view and layer controller
        vc.link_views_by_dict(all_views, {
            "spatialTargetZ": 0,
            "spatialTargetT": 0,
            "imageLayer": CL([{
                "fileUid": 'visium',
                "spatialLayerOpacity": 1,
                "spatialLayerVisible": True,
                "photometricInterpretation": 'RGB',
            }]),
            "spotLayer": CL([{
                "obsType": 'spot',
                "spatialLayerVisible": True,
                "spatialLayerOpacity": 0.5,
                "spatialSpotRadius": self._get_scale_factor(),
                "featureValueColormapRange": [0, 1],
            }]),
        })

        # Indicate obs type for all views
        vc.link_views(spot_views, ['obsType'], ['spot'])

        return vc
