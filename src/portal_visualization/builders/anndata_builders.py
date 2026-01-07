from functools import cached_property

import numpy as np
import zarr
from vitessce import (
    AnnDataWrapper,
    ImageOmeTiffWrapper,
    MultivecZarrWrapper,
    SpatialDataWrapper,
    VitessceConfig,
    VitessceConfigDatasetFile,
    get_initial_coordination_scope_prefix,
)
from vitessce import Component as cm
from vitessce import CoordinationLevel as CL
from vitessce import CoordinationType as ct
from vitessce import ViewType as vt

from ..constants import MAX_OBS_FOR_HEATMAP, MULTIOMIC_ZARR_PATH, XENIUM_ZARR_PATH, ZARR_PATH, ZIP_ZARR_PATH
from ..utils import get_conf_cells, obs_has_column, read_zip_zarr
from .base_builders import ViewConfBuilder

RNA_SEQ_ANNDATA_FACTOR_PATHS = [
    f"obs/{key}" for key in ["marker_gene_0", "marker_gene_1", "marker_gene_2", "marker_gene_3", "marker_gene_4"]
]

RNA_SEQ_FACTOR_LABEL_NAMES = [f"Marker Gene {i}" for i in range(len(RNA_SEQ_ANNDATA_FACTOR_PATHS))]


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
        self._is_zarr_zip = False
        self._spatial_w = 0
        self._obs_set_paths = None
        self._obs_set_names = None
        self._obs_labels_paths = None
        self._obs_labels_names = None
        self._marker = None
        self._gene_alias = None
        self._views = None
        self._is_annotated = None
        self._scatterplot_w = None
        self._scatterplot_h = None

    @cached_property
    def zarr_store(self):
        request_init = self._get_request_init() or {}
        zarr_path = ZIP_ZARR_PATH if self._is_zarr_zip else ZARR_PATH

        if self._is_zarr_zip:
            zarr_url = self._build_assets_url(zarr_path, use_token=True)
            try:
                return read_zip_zarr(zarr_url, request_init)
            except Exception as e:  # pragma: no cover
                print(f"Error opening the zip zarr file. {e}")
                return None
        else:
            zarr_url = self._build_assets_url(zarr_path, use_token=False)
            return zarr.open(zarr_url, mode="r", storage_options={"client_kwargs": request_init})

    @cached_property
    def has_marker_genes(self):
        z = self.zarr_store
        if z is not None and "obs/marker_gene_0" in z:
            return True

    @cached_property
    def is_annotated(self):
        z = self.zarr_store
        if z is not None and "uns/annotation_metadata/is_annotated" in z:
            return z["uns/annotation_metadata/is_annotated"][()]
        else:
            return False

    @cached_property
    def n_obs(self):
        """Get the number of observations in the dataset.

        >>> from pathlib import Path
        >>> import json
        >>> import zarr
        >>> fixture_path = Path(__file__).parent.parent.parent.parent / "test" / "good-fixtures" / "RNASeqAnnDataZarrViewConfBuilder" / "fake-is-not-annotated-published-entity.json"
        >>> entity = json.loads(fixture_path.read_text())
        >>> builder = RNASeqAnnDataZarrViewConfBuilder(entity, 'token', 'https://example.com')
        >>> # Mock zarr store with obs index
        >>> z = zarr.open_group()
        >>> obs_group = z.create_group('obs')
        >>> obs_group['_index'] = zarr.array(['cell_0', 'cell_1', 'cell_2'])
        >>> # Set the cached property value directly on the instance
        >>> builder.__dict__['zarr_store'] = z
        >>> builder.n_obs
        3
        """
        z = self.zarr_store
        if z is not None and "obs" in z:
            obs = z["obs"]
            # obs is a zarr Group containing obs annotations
            # The _index array contains the observation identifiers
            if "_index" in obs:
                # Standard anndata zarr structure
                return obs["_index"].shape[0]
            # Fallback: try to get from any obs column
            for key in obs:  # pragma: no cover
                if hasattr(obs[key], "shape"):
                    return obs[key].shape[0]
        return 0  # pragma: no cover

    def compute_scatterplot_w(self):
        return 6

    def compute_scatterplot_h(self):
        return 12 if self._minimal else 6

    def _should_include_optional_views(self, view_type=None):
        """Determine if optional views should be included based on minimal flag and dataset size.

        For heatmap views, also check if the dataset has too many observations for performance.

        >>> from pathlib import Path
        >>> import json
        >>> fixture_path = Path(__file__).parent.parent.parent.parent / "test" / "good-fixtures" / "RNASeqAnnDataZarrViewConfBuilder" / "fake-is-not-annotated-published-entity.json"
        >>> entity = json.loads(fixture_path.read_text())
        >>> builder = RNASeqAnnDataZarrViewConfBuilder(entity, 'token', 'https://example.com')
        >>> # Test with small dataset - set cached n_obs value directly on instance
        >>> builder._minimal = False
        >>> builder.__dict__['n_obs'] = 50000
        >>> builder._should_include_optional_views('heatmap')
        True
        >>> # Test with large dataset
        >>> builder.__dict__['n_obs'] = 150000
        >>> builder._should_include_optional_views('heatmap')
        False
        >>> # Test non-heatmap view with large dataset
        >>> builder._should_include_optional_views('gene_list')
        True
        >>> # Test minimal mode
        >>> builder._minimal = True
        >>> builder.__dict__['n_obs'] = 50000
        >>> builder._should_include_optional_views('heatmap')
        False
        """

        if self._minimal:
            return False
        return not (view_type == "heatmap" and self.n_obs > MAX_OBS_FOR_HEATMAP)

    def get_conf_cells(self, marker=None):
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        # Use .zgroup file as proxy for whether or not the zarr store is present.
        if f"{ZARR_PATH}.zip" in file_paths_found:
            self._is_zarr_zip = True
        elif f"{ZARR_PATH}/.zgroup" not in file_paths_found:
            message = f"RNA-seq assay with uuid {self._uuid} has no .zarr store at {ZARR_PATH}"
            raise FileNotFoundError(message)
        self._is_annotated = self.is_annotated
        if self._scatterplot_w is None:
            self._scatterplot_w = self.compute_scatterplot_w()
        if self._scatterplot_h is None:
            self._scatterplot_h = self.compute_scatterplot_h()
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
        gene_alias = "var/hugo_symbol" if z is not None and "var" in z and "hugo_symbol" in z["var"] else None
        if gene_alias is not None and marker is not None:
            # If user has indicated a marker gene in parameters and we have a hugo_symbol mapping,
            # then we need to convert it to the proper underlying ensembl ID for the dataset
            # in order for the views to reflect the correct gene.

            obs_attrs = z["obs"].attrs.asdict()
            encoding_version = obs_attrs["encoding-version"]

            # Encoding Version 0.1.0
            # https://anndata.readthedocs.io/en/0.7.8/fileformat-prose.html#categorical-arrays
            if encoding_version == "0.1.0":
                # Get the list of ensembl IDs from the zarr store
                ensembl_ids_key = z["var"].attrs["_index"]
                ensembl_ids = z["var"][ensembl_ids_key]
                # Get the list of hugo symbols
                hugo_symbols = z[gene_alias]
                # Indices in hugo_index_list match to indices of mapped ensembl ID's
                # Values in hugo_index_list match to indices in hugo_categories
                hugo_index_list = hugo_symbols[:]
                # Get the key for the hugo categories list from the categorical entry attributes
                hugo_categories_key = hugo_symbols.attrs.asdict()["categories"]
                # Get the list of categories that the hugo_index_list's values map to
                hugo_categories = z["var"][hugo_categories_key][:]
                # Find the index of the user-provided marker gene in the list of hugo symbols
                marker_index_in_categories = np.where(hugo_categories == marker)[0][0]
                # If the user-provided gene's index is found, continue
                if marker_index_in_categories >= 0:
                    # Find index of HUGO pointer corresponding to marker gene
                    marker_index = np.where(hugo_index_list == marker_index_in_categories)[0][0]
                    # If valid index is found, set the marker name to the corresponding Ensembl ID
                    if marker_index >= 0:
                        marker = ensembl_ids[marker_index]
                    else:
                        pass  # pragma: no cover
            # Encoding Version 0.2.0
            # https://anndata.readthedocs.io/en/latest/fileformat-prose.html#categorical-arrays
            # Our pipeline currently does not use this encoding version
            # Future improvement to be implemented in CAT-137
            # elif (encoding_version == "0.2.0"):
            #     print('TODO - Encoding Version 0.2.0 support')
        self._marker = str(marker) if marker is not None else None
        self._gene_alias = gene_alias

    def _set_up_dataset(self, vc):
        zarr_path = ZIP_ZARR_PATH if self._is_zarr_zip else ZARR_PATH
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        z = self.zarr_store
        dataset = vc.add_dataset(name=self._uuid).add_object(
            AnnDataWrapper(
                adata_url=adata_url,
                is_zip=self._is_zarr_zip,
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
                feature_labels_path="var/hugo_symbol" if z is not None and "var" in z else None,
                gene_alias=self._gene_alias,
                obs_labels_paths=self._obs_labels_paths,
                obs_labels_names=self._obs_labels_names,
            )
        )
        return dataset

    def _set_up_obs_labels(
        self,
        additional_obs_labels_paths=[],
        additional_obs_labels_names=[],
        additional_obs_set_paths=[],
        additional_obs_set_names=[],
        # Optionally skip default obs paths and labels
        skip_default_paths=False,
        # Support multiomic datasets
        modality_prefix=None,
    ):
        # Some of the keys (like marker_genes_for_heatmap) here are from our pipeline
        # https://github.com/hubmapconsortium/portal-containers/blob/master/containers/anndata-to-ui
        # while others come from Matt's standard scanpy pipeline
        # or AnnData default (like X_umap or X).
        # obs sets are annotated sets of cells/clusters used to color the cells
        obs_set_paths = []
        obs_set_names = []
        # obs labels are tooltip helpers which e.g. identify highly expressed genes
        # or help map predicted cell labels to their IDs
        obs_label_paths = []
        obs_label_names = []

        # Add additional obs labels and sets if provided
        obs_set_paths.extend(additional_obs_set_paths)
        obs_set_names.extend(additional_obs_set_names)
        obs_label_paths.extend(additional_obs_labels_paths)
        obs_label_names.extend(additional_obs_labels_names)

        z = self.zarr_store
        obs = None if z is None else z["obs"] if modality_prefix is None else z[f"{modality_prefix}/obs"]
        if not skip_default_paths:
            if self._is_annotated:
                azimuth_categories = self._get_azimuth_categories(obs)
                if "predicted.ASCT.celltype" in obs:
                    obs_set_paths.append("obs/predicted.ASCT.celltype")
                    obs_set_names.append("Predicted ASCT Cell Type")
                if "predicted_label" in obs:
                    obs_set_paths.append("obs/predicted_label")
                    obs_set_names.append("Cell Ontology Annotation")
                if "predicted_CLID" in obs:
                    obs_label_paths.append("obs/predicted_CLID")
                    obs_label_names.append("Predicted CL ID")
                if "CL_Label" in obs:
                    obs_set_paths.append("obs/CL_Label")
                    obs_set_names.append("CL Label")
                if len(azimuth_categories) > 0:
                    obs_set_paths.append(azimuth_categories)
                    obs_set_names.append("Azimuth Categories")
                if "final_level_labels" in obs:
                    obs_set_paths.append("obs/final_level_labels")
                    obs_set_names.append("Final Level Labels")
                if "full_hierarchical_labels" in obs:
                    obs_set_paths.append("obs/full_hierarchical_labels")
                    obs_set_names.append("Full Hierarchical Labels")

            obs_set_paths.append("obs/leiden")
            obs_set_names.append("Leiden")
        if self.has_marker_genes:
            obs_label_paths.extend(RNA_SEQ_ANNDATA_FACTOR_PATHS)
            obs_label_names.extend(RNA_SEQ_FACTOR_LABEL_NAMES)

        self._obs_set_paths = obs_set_paths
        self._obs_set_names = obs_set_names
        self._obs_labels_paths = obs_label_paths
        self._obs_labels_names = obs_label_names

    def _setup_anndata_view_config(self, vc: VitessceConfig, dataset: VitessceConfigDatasetFile):
        scatterplot = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP")

        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset)

        gene_list = None
        heatmap = None
        if self._should_include_optional_views("gene_list"):
            gene_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
        if self._should_include_optional_views("heatmap"):
            heatmap = vc.add_view(cm.HEATMAP, dataset=dataset)

        cell_sets_expr = vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset)

        # Spatial view is added if present, otherwise gets filtered out before views are linked
        # This ensures that the view config is valid for datasets with and without a spatial view
        spatial = self._add_spatial_view(dataset, vc)
        views = list(
            filter(
                lambda v: v is not None,
                [  # pragma: no cover
                    cell_sets,
                    gene_list,
                    scatterplot,
                    cell_sets_expr,
                    heatmap,
                    spatial,
                ],
            )
        )

        # Handle layout for variants in one unified place
        if self._minimal:
            if self._is_spatial:
                scatterplot.set_xywh(x=0, y=0, w=6, h=6)
                spatial.set_xywh(x=0, y=6, w=6, h=6)
                cell_sets.set_xywh(x=6, y=0, w=6, h=4)
                cell_sets_expr.set_xywh(x=6, y=4, w=6, h=8)
                self._views = [scatterplot, spatial, cell_sets_expr]
            else:
                scatterplot.set_xywh(x=0, y=0, w=6, h=12)
                cell_sets.set_xywh(x=6, y=0, w=6, h=4)
                cell_sets_expr.set_xywh(x=6, y=4, w=6, h=8)

                self._views = [scatterplot, cell_sets_expr]
        else:
            if self._is_spatial:
                if heatmap is not None:
                    vc.layout(((scatterplot | spatial) / heatmap) | ((cell_sets | gene_list) / cell_sets_expr))
                else:
                    # When heatmap is hidden, expand scatterplot/spatial vertically to fill the space
                    vc.layout((scatterplot | spatial) | ((cell_sets | gene_list) / cell_sets_expr))
            else:
                if heatmap is not None:
                    vc.layout((scatterplot / heatmap) | ((cell_sets | gene_list) / cell_sets_expr))
                else:
                    # When heatmap is hidden, expand scatterplot vertically to fill the space
                    vc.layout(scatterplot | ((cell_sets | gene_list) / cell_sets_expr))
                    scatterplot.set_xywh(x=0, y=0, w=6, h=12)
            # Adjust the cell sets and gene list to not be as tall,
            # give cell sets expression more height
            cell_sets.set_xywh(x=6, y=0, w=3, h=4)
            if gene_list is not None:
                gene_list.set_xywh(x=9, y=0, w=3, h=4)
            cell_sets_expr.set_xywh(x=6, y=4, w=6, h=8)
            self._views = views

        return vc

    def _add_spatial_view(self, dataset, vc):
        # This class does not have a spatial_view...
        # but the subclass does, and overrides this method.
        return None

    def _link_marker_gene(self, vc):
        # Link top 5 marker genes
        vc.link_views(
            self._views,
            [ct.OBS_LABELS_TYPE for _ in self._obs_labels_names],
            self._obs_labels_names,
            allow_multiple_scopes_per_type=True,
        )
        # Link user-provided marker gene
        if self._marker:
            vc.link_views(
                self._views,
                [ct.FEATURE_SELECTION, ct.OBS_COLOR_ENCODING],
                [[self._marker], "geneSelection"],
            )
        return vc

    def _get_azimuth_categories(self, obs):
        azimuth_categories = []
        if "azimuth_broad" in obs:
            azimuth_categories.append("obs/azimuth_broad")
        if "azimuth_medium" in obs:
            azimuth_categories.append("obs/azimuth_medium")
        if "azimuth_fine" in obs:
            azimuth_categories.append("obs/azimuth_fine")
        return azimuth_categories


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
        self._scatterplot_h = self.compute_scatterplot_h()

    def _add_spatial_view(self, dataset, vc):
        spatial = vc.add_view(
            cm.SPATIAL, dataset=dataset, x=self._scatterplot_w, y=0, w=self._spatial_w, h=self._scatterplot_h
        )
        [cells_layer] = vc.add_coordination("spatialSegmentationLayer")
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

    def _set_visium_datasets(self, vc, image_url, offsets_url, adata_url):
        visium_image = ImageOmeTiffWrapper(
            img_url=image_url,
            uid=self._uuid,
            offsets_url=offsets_url,
            request_init=self._get_request_init(),
        )
        visium_spots = AnnDataWrapper(
            adata_url=adata_url,
            iz_zip=self._is_zarr_zip,
            obs_feature_matrix_path="X",
            obs_set_paths=self._obs_set_paths,
            obs_set_names=self._obs_set_names,
            obs_labels_names=self._obs_labels_names,
            obs_labels_paths=self._obs_labels_paths,
            obs_spots_path="obsm/X_spatial",
            obs_embedding_paths=["obsm/X_umap", "obsm/X_pca"],
            obs_embedding_names=["UMAP", "PCA"],
            obs_embedding_dims=[[0, 1], [0, 1]],
            feature_labels_path="var/hugo_symbol",
            request_init=self._get_request_init(),
            initial_feature_filter_path="var/top_highly_variable",
            coordination_values={
                "obsType": "spot",
            },
        )
        dataset = vc.add_dataset(name="Visium", uid=self._uuid).add_object(visium_image).add_object(visium_spots)
        return dataset

    def _set_visium_config(self, vc, dataset):
        # Add / lay out views
        # Conditionally add heatmap based on dataset size
        heatmap = None
        if self._should_include_optional_views("heatmap"):
            # Standard layout with heatmap
            umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", w=3, h=6, x=0, y=0)
            spatial = vc.add_view("spatialBeta", dataset=dataset, w=3, h=6, x=3, y=0)
            heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, w=6, h=6, x=0, y=6).set_props(transpose=True)
        else:
            # Expanded layout without heatmap - extend views to fill vertical space
            umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", w=3, h=12, x=0, y=0)
            spatial = vc.add_view("spatialBeta", dataset=dataset, w=3, h=12, x=3, y=0)

        lc = vc.add_view("layerControllerBeta", dataset=dataset, w=6, h=3, x=6, y=0)

        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset, w=3, h=4, x=6, y=2)

        gene_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset, w=3, h=4, x=9, y=2)

        cell_sets_expr = vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset, w=3, h=5, x=6, y=7)

        cell_set_sizes = vc.add_view(cm.OBS_SET_SIZES, dataset=dataset, w=3, h=5, x=9, y=7)

        all_views = list(
            filter(
                lambda v: v is not None,
                [spatial, lc, umap, cell_sets, cell_sets_expr, gene_list, cell_set_sizes, heatmap],
            )
        )

        self._views = all_views
        spatial_views = [spatial, lc]

        # selected_gene_views = [umap, gene_list, heatmap, spatial]

        # Indicate obs type for all views
        vc.link_views(all_views, ["obsType"], ["spot"])
        vc.link_views_by_dict(
            spatial_views,
            {
                "imageLayer": CL(
                    [
                        {
                            "photometricInterpretation": self._photometricInterpretation,
                        }
                    ]
                ),
            },
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "image"),
        )
        vc.link_views_by_dict(
            spatial_views,
            {
                "spotLayer": CL(
                    [
                        {
                            "spatialLayerOpacity": 1,
                            "spatialSpotRadius": self._get_spot_radius(),
                        }
                    ]
                ),
            },
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "obsSpots"),
        )
        return vc


class SpatialMultiomicAnnDataZarrViewConfBuilder(SpatialRNASeqAnnDataZarrViewConfBuilder):
    """
    Wrapper class for creating a AnnData-backed view configuration for multiomic spatial data
    such as Visium.
    Example: https://portal.hubmapconsortium.org/browse/dataset/a9335618873a33cd060803c82bdefe89
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._scatterplot_w = 3
        self._spatial_w = 3
        self._photometricInterpretation = "RGB"

    def _get_spot_radius(self):
        z = self.zarr_store
        visium_scalefactor_path = "spatial/visium/scalefactors/spot_diameter_micrometers"
        if visium_scalefactor_path in z["uns"]:
            # Since the scale factor is the diameter, we divide by 2 to get the radius
            return z["uns"][visium_scalefactor_path][()].tolist() / 2

    def _set_up_dataset(self, vc):
        file_paths_found = self._get_file_paths()
        zarr_path = ZARR_PATH
        if any(".zarr.zip" in path for path in file_paths_found):  # pragma: no cover
            self._is_zarr_zip = True
            zarr_path = ZIP_ZARR_PATH

        elif f"{ZARR_PATH}/.zgroup" not in file_paths_found:  # pragma: no cover
            message = f"RNA-seq assay with uuid {self._uuid} has no .zarr store at {ZARR_PATH}"
            raise FileNotFoundError(message)
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        image_url = self._build_assets_url("ometiff-pyramids/visium_histology_hires_pyramid.ome.tif", use_token=True)
        offsets_url = self._build_assets_url(
            "output_offsets/visium_histology_hires_pyramid.offsets.json", use_token=True
        )

        # Add dataset with Visium image and secondary analysis anndata

        dataset = self._set_visium_datasets(vc, image_url, offsets_url, adata_url)
        return dataset

    def _setup_anndata_view_config(self, vc, dataset):
        return self._set_visium_config(vc, dataset)


class XeniumMultiomicAnnDataZarrViewConfBuilder(SpatialRNASeqAnnDataZarrViewConfBuilder):
    """
    Wrapper class for creating a AnnData-backed view configuration for multiomic spatial data
    such as Visium.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._scatterplot_w = 3
        self._spatial_w = 3
        self._photometricInterpretation = "BlackIsZero"
        self._is_spatial_zarr_zip = False

    def _set_up_dataset(self, vc):
        file_paths_found = self._get_file_paths()
        adata_url = self._add_zarr_files(ZARR_PATH, file_paths_found)
        spatial_data_url = self._add_zarr_files(XENIUM_ZARR_PATH, file_paths_found)

        dataset = self._set_xenium_datasets(vc, adata_url, spatial_data_url)
        return dataset

    def _setup_anndata_view_config(self, vc, dataset):
        return self._set_xenium_config(vc, dataset)

    def _add_zarr_files(self, zarr_path, file_paths_found):
        if any(f"{zarr_path}.zip" in path for path in file_paths_found):  # pragma: no cover
            if "xenium" in zarr_path.lower():
                self._is_spatial_zarr_zip = True
            else:
                self._is_zarr_zip = True
            zarr_path = f"{zarr_path}.zip"

        elif (
            self._is_zarr_zip is False or self._is_spatial_zarr_zip is False
        ) and f"{zarr_path}/.zgroup" not in file_paths_found:  # pragma: no cover
            message = f"RNA-seq assay with uuid {self._uuid} has no .zarr store at {zarr_path}"
            raise FileNotFoundError(message)
        return self._build_assets_url(zarr_path, use_token=False)

    def _set_xenium_datasets(self, vc, adata_url, spatial_data_url):
        self._set_up_obs_labels()
        spatial_data = SpatialDataWrapper(
            sdata_url=spatial_data_url,
            is_zip=self._is_spatial_zarr_zip,
            table_path="tables/table",
            image_path="images/morphology_focus",
            obs_segmentations_path="labels/cell_labels",
            request_init=self._get_request_init(),
            coordination_values={"obsType": "spot"},
        )
        visium_spots = AnnDataWrapper(
            adata_url=adata_url,
            iz_zip=self._is_zarr_zip,
            obs_feature_matrix_path="X",
            obs_set_paths=self._obs_set_paths,
            obs_set_names=self._obs_set_names,
            obs_labels_names=self._obs_labels_names,
            obs_labels_paths=self._obs_labels_paths,
            obs_embedding_paths=["obsm/X_umap"],
            obs_embedding_names=["UMAP"],
            request_init=self._get_request_init(),
            coordination_values={"obsType": "spot"},
        )
        dataset = vc.add_dataset(name="Xenium", uid=self._uuid).add_object(spatial_data).add_object(visium_spots)
        return dataset

    def _set_xenium_config(self, vc, dataset):
        [obs_color_encoding_scope] = vc.add_coordination("obsColorEncoding")
        obs_color_encoding_scope.set_value("cellSetSelection")

        # Conditionally add heatmap based on dataset size
        heatmap = None
        if self._should_include_optional_views("heatmap"):
            # Standard layout with heatmap
            umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", w=3, h=6, x=0, y=0)
            spatial = vc.add_view("spatialBeta", dataset=dataset, w=3, h=6, x=3, y=0)
            heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, w=6, h=6, x=0, y=6).set_props(transpose=True)
        else:
            # Expanded layout without heatmap - extend views to fill vertical space
            umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", w=3, h=12, x=0, y=0)
            spatial = vc.add_view("spatialBeta", dataset=dataset, w=3, h=12, x=3, y=0)

        lc = vc.add_view("layerControllerBeta", dataset=dataset, w=6, h=3, x=6, y=0)

        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset, w=3, h=4, x=6, y=2)

        cell_sets.use_coordination(obs_color_encoding_scope)

        gene_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset, w=3, h=4, x=9, y=2)

        cell_sets_expr = vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset, w=3, h=5, x=6, y=7)

        cell_set_sizes = vc.add_view(cm.OBS_SET_SIZES, dataset=dataset, w=3, h=5, x=9, y=7)

        all_views = list(
            filter(
                lambda v: v is not None,
                [spatial, lc, umap, cell_sets, cell_sets_expr, gene_list, cell_set_sizes, heatmap],
            )
        )

        self._views = all_views
        vc.link_views(all_views, ["obsType"], ["spot"])

        vc.link_views_by_dict(
            [spatial, lc],
            {
                "spatialTargetZ": 0,
                "spatialTargetT": 0,
                "imageLayer": CL(
                    [
                        {
                            "photometricInterpretation": "BlackIsZero",
                        }
                    ]
                ),
            },
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "image"),
        )

        vc.link_views_by_dict(
            [spatial, lc],
            {"segmentationLayer": CL([{"segmentationChannel": CL([{"obsColorEncoding": obs_color_encoding_scope}])}])},
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "obsSegmentations"),
        )

        return vc


class MultiomicAnndataZarrViewConfBuilder(RNASeqAnnDataZarrViewConfBuilder):
    """Wrapper class for creating a AnnData-backed view configuration
    for multiomic data from mudata-to-ui.cwl like 10X Multiome
    Example dataset:
    https://portal.hubmapconsortium.org/browse/dataset/024d671f28994ff76eebf1e24ee640a7
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._scatterplot_w = 3

    @cached_property
    def zarr_store(self):
        zarr_path = f"{MULTIOMIC_ZARR_PATH}.zip" if self._is_zarr_zip else MULTIOMIC_ZARR_PATH
        request_init = self._get_request_init() or {}
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        return zarr.open(adata_url, mode="r", storage_options={"client_kwargs": request_init})

    @cached_property
    def has_marker_genes(self):
        z = self.zarr_store
        return "mod/rna/var/marker_genes_for_heatmap" in z

    @cached_property
    def has_cbb(self):
        z = self.zarr_store
        return "mod/atac_cbb" in z

    @cached_property
    def is_annotated(self):
        z = self.zarr_store
        if "mod/rna/uns/annotation_metadata/is_annotated" in z:
            return z["mod/rna/uns/annotation_metadata/is_annotated"][()]
        else:
            return False

    @cached_property
    def n_obs(self):
        """Get the number of observations in the multiomics dataset.

        >>> from pathlib import Path
        >>> import json
        >>> import zarr
        >>> fixture_path = Path(__file__).parent.parent.parent.parent / "test" / "good-fixtures" / "MultiomicAnndataZarrViewConfBuilder" / "fake-multiome-entity.json"
        >>> entity = json.loads(fixture_path.read_text())
        >>> builder = MultiomicAnndataZarrViewConfBuilder(entity, 'token', 'https://example.com')
        >>> # Case 1: _index array exists (most common case)
        >>> z = zarr.open_group()
        >>> rna_mod = z.create_group('mod/rna')
        >>> obs_group = rna_mod.create_group('obs')
        >>> obs_group['_index'] = zarr.array(['cell_0', 'cell_1', 'cell_2'])
        >>> builder.__dict__['zarr_store'] = z
        >>> builder.n_obs
        3
        >>> # Case 2: fallback to first key with shape
        >>> z2 = zarr.open_group()
        >>> rna_mod2 = z2.create_group('mod/rna')
        >>> obs_group2 = rna_mod2.create_group('obs')
        >>> obs_group2['leiden'] = zarr.array(['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3'])
        >>> builder2 = MultiomicAnndataZarrViewConfBuilder(entity, 'token', 'https://example.com')
        >>> builder2.__dict__['zarr_store'] = z2
        >>> builder2.n_obs
        4
        >>> # Case 3: no obs data, return 0
        >>> z3 = zarr.open_group()
        >>> builder3 = MultiomicAnndataZarrViewConfBuilder(entity, 'token', 'https://example.com')
        >>> builder3.__dict__['zarr_store'] = z3
        >>> builder3.n_obs
        0
        """
        z = self.zarr_store
        if z is not None and "mod/rna/obs" in z:
            obs = z["mod/rna/obs"]
            if "_index" in obs:
                return obs["_index"].shape[0]
            # Fallback: try to get from any obs column
            for key in obs:
                if hasattr(obs[key], "shape"):
                    return obs[key].shape[0]
        return 0

    def get_conf_cells(self, marker=None):
        modality_prefix = "mod/rna/obs"
        z = self.zarr_store
        obs = None if z is None else z[modality_prefix]
        # file_paths_found = [file["rel_path"] for file in self._entity["files"] if "files" in self._entity]
        # # Use .zgroup file as proxy for whether or not the zarr store is present.
        # if any('.zarr.zip' in path for path in file_paths_found): # pragma: no cover
        #     self._is_zarr_zip = True
        # elif not self._is_zarr_zip and f'{MULTIOMIC_ZARR_PATH}/.zgroup' not in file_paths_found:  # pragma: no cover
        #     message = f'Multiomic assay with uuid {self._uuid} has no .zarr store at {MULTIOMIC_ZARR_PATH}'
        #     raise FileNotFoundError(message)

        # Each clustering has its own genomic profile; since we can't currently toggle between
        # selected genomic profiles, each clustering needs its own view config.
        self._is_annotated = self.is_annotated
        confs = []
        cluster_columns = [
            ["leiden_wnn", "Leiden (Weighted Nearest Neighbor)", "wnn"],
            ["cluster_atac", "ArchR Clusters (ATAC)", "cbb"] if self.has_cbb else None,
            ["leiden_rna", "Leiden (RNA)", "rna"],
            ["predicted_label", "Cell Ontology Annotation", "label"] if self._is_annotated else None,
            ["full_hierarchical_labels", "Full Hierarchical Labels", "label"] if self._is_annotated else None,
            ["final_level_labels", "Final Level Labels", "label"] if self._is_annotated else None,
            ["CL_Label", "CL Label", "label"] if self._is_annotated else None,
        ]

        cluster_columns = [
            col for col in cluster_columns if col is not None and obs_has_column(z, col[0], modality_prefix)
        ]

        column_names, column_labels = [f"obs/{col[0]}" for col in cluster_columns], [col[1] for col in cluster_columns]

        azimuth_categories = self._get_azimuth_categories(obs)

        if len(azimuth_categories) > 0:
            column_names.append(azimuth_categories)
            column_labels.append("Azimuth Categories")

        self._set_up_marker_gene(marker)
        self._set_up_obs_labels(
            additional_obs_set_names=column_labels,
            additional_obs_set_paths=column_names,
            skip_default_paths=True,
            modality_prefix="mod/rna",
        )

        for column_name, column_label, multivec_label in cluster_columns:
            vc = VitessceConfig(name=f"{column_label}", schema_version=self._schema_version)
            dataset = self._set_up_dataset(vc, multivec_label)
            vc = self._setup_anndata_view_config(vc, dataset, column_name, column_label)
            vc = self._link_marker_gene(vc)
            confs.append(vc.to_dict())
        return get_conf_cells(confs)

    def _set_up_dataset(self, vc, multivec_label):
        zarr_base = "hubmap_ui/mudata-zarr"
        zarr_path = f"{zarr_base}/secondary_analysis.zarr"
        h5mu_zarr = self._build_assets_url(zarr_path, use_token=False)
        rna_zarr = self._build_assets_url(f"{zarr_path}/mod/rna", use_token=False)
        atac_cbg_zarr = self._build_assets_url(f"{zarr_path}/mod/atac_cbg", use_token=False)
        multivec_zarr = self._build_assets_url(f"{zarr_base}/{multivec_label}.multivec.zarr", use_token=False)
        dataset = (
            vc.add_dataset(name=multivec_label)
            .add_object(
                MultivecZarrWrapper(
                    zarr_url=multivec_zarr,
                    request_init=self._get_request_init(),
                )
            )
            .add_object(
                AnnDataWrapper(
                    # We run add_object with adata_path=rna_zarr first to add the cell-by-gene
                    # matrix and associated metadata.
                    adata_url=rna_zarr,
                    # is_zip=self._is_zarr_zip,
                    obs_embedding_paths=["obsm/X_umap"],
                    obs_embedding_names=["UMAP - RNA"],
                    obs_set_paths=self._obs_set_paths,
                    obs_set_names=self._obs_set_names,
                    obs_feature_matrix_path="X",
                    initial_feature_filter_path="var/highly_variable",
                    feature_labels_path="var/hugo_symbol",
                    request_init=self._get_request_init(),
                    # To be explicit that the features represent genes and gene expression, we
                    # specify that here.
                    coordination_values={
                        "featureType": "gene",
                        "featureValueType": "expression",
                        "featureLabelsType": "gene",
                    },
                )
            )
            .add_object(
                AnnDataWrapper(
                    adata_url=atac_cbg_zarr,
                    obs_feature_matrix_path="X",
                    initial_feature_filter_path="var/highly_variable",
                    obs_embedding_paths=["obsm/X_umap"],
                    obs_embedding_names=["UMAP - ATAC"],
                    request_init=self._get_request_init(),
                    # To be explicit that the features represent genes and gene expression, we
                    # specify that here.
                    coordination_values={
                        "featureType": "peak",
                        "featureValueType": "count",
                    },
                )
            )
            .add_object(
                AnnDataWrapper(
                    adata_url=h5mu_zarr,
                    # is_zip=self._is_zarr_zip,
                    obs_feature_matrix_path="X",
                    obs_embedding_paths=["obsm/X_umap"],
                    obs_embedding_names=["UMAP - WNN"],
                    request_init=self._get_request_init(),
                    coordination_values={"featureType": "other"},
                )
            )
        )
        return dataset

    def _setup_anndata_view_config(self, vc, dataset, column_name, column_label):
        umap_scatterplot_by_rna = vc.add_view(vt.SCATTERPLOT, dataset=dataset, mapping="UMAP - RNA").set_props(
            embeddingCellSetLabelsVisible=False
        )
        umap_scatterplot_by_atac = vc.add_view(vt.SCATTERPLOT, dataset=dataset, mapping="UMAP - ATAC").set_props(
            embeddingCellSetLabelsVisible=False
        )
        umap_scatterplot_by_wnn = vc.add_view(vt.SCATTERPLOT, dataset=dataset, mapping="UMAP - WNN").set_props(
            embeddingCellSetLabelsVisible=False
        )

        gene_list = vc.add_view(vt.FEATURE_LIST, dataset=dataset)
        peak_list = vc.add_view(vt.FEATURE_LIST, dataset=dataset)

        # rna_heatmap = vc.add_view(vt.HEATMAP, dataset=dataset).set_props(transpose=False)
        # atac_heatmap = vc.add_view(vt.HEATMAP, dataset=dataset).set_props(transpose=False)
        genomic_profiles = vc.add_view(vt.GENOMIC_PROFILES, dataset=dataset)
        genomic_profiles.set_props(title=f"{column_label} Genomic Profiles")

        cell_sets = vc.add_view(vt.OBS_SETS, dataset=dataset)

        # specify which of the two features' (i.e., genes or peaks) views correspond to
        # We also need to make sure the selection of genes and peaks are scoped only to
        # the corresponding view,
        # and we want to make sure the color mappings are independent for each modality.
        coordination_types = [ct.FEATURE_TYPE, ct.FEATURE_VALUE_TYPE]
        vc.link_views([umap_scatterplot_by_rna, gene_list], coordination_types, ["gene", "expression"])
        vc.link_views([umap_scatterplot_by_atac, peak_list], coordination_types, ["peak", "count"])

        # Coordinate the selection of cell sets between the scatterplots and lists
        # of features/observations.
        coordination_types = [
            ct.FEATURE_SELECTION,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_VALUE_COLORMAP_RANGE,
            ct.OBS_SET_SELECTION,
        ]

        label_names = self._get_obs_set_members(column_name)
        obs_set_coordinations = [[column_label, str(i)] for i in label_names]
        vc.link_views(
            [
                umap_scatterplot_by_rna,
                umap_scatterplot_by_atac,
                umap_scatterplot_by_wnn,
                gene_list,
                peak_list,
                cell_sets,
            ],
            coordination_types,
            [None, "cellSetSelection", [0.0, 1.0], obs_set_coordinations],
        )

        # Indicate genomic profiles' clusters; based on the display name for the ATAC CBB clusters.
        obs_set_coordination, obs_color_coordination = vc.add_coordination(ct.OBS_SET_SELECTION, ct.OBS_COLOR_ENCODING)
        genomic_profiles.use_coordination(obs_set_coordination, obs_color_coordination)
        obs_set_coordination.set_value(obs_set_coordinations)
        obs_color_coordination.set_value("cellSetSelection")

        # Hide numeric cluster labels
        vc.link_views(
            [umap_scatterplot_by_rna, umap_scatterplot_by_atac, umap_scatterplot_by_wnn],
            [ct.EMBEDDING_OBS_SET_LABELS_VISIBLE],
            [False],
        )

        vc.layout(
            ((umap_scatterplot_by_rna | umap_scatterplot_by_atac) | (umap_scatterplot_by_wnn | cell_sets))
            / (genomic_profiles | (peak_list | gene_list))
        )

        self._views = [
            umap_scatterplot_by_rna,
            umap_scatterplot_by_atac,
            umap_scatterplot_by_wnn,
            gene_list,
            peak_list,
            genomic_profiles,
            cell_sets,
        ]
        return vc

    def _get_obs_set_members(self, column_name):
        z = self.zarr_store
        members = z[f"mod/rna/obs/{column_name}"].categories
        return members
