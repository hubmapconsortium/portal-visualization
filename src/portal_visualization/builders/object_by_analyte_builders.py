from functools import cached_property

from vitessce import AnnDataWrapper, VitessceConfig
from vitessce import Component as cm

from ..utils import get_conf_cells, read_zip_zarr
from .base_builders import ViewConfBuilder

"""
# Object by Analyte EPIC builders contain fully self-contained data
# across a variety of modalities, all within one mudata file.
# These files have been consistently transformed using zarr zip format.
# Since these files represent independent analyses and are not the product of
# standardized HuBMAP analyses, the visualizations are more dynamic and depend on
# additional metadata provided in a json file.

# Object by analyte specification:
# https://docs.google.com/document/d/1TkmleE99wpynqSa0MS47Z8Q2vG1ru47fNFl-5KFJKoo/edit?tab=t.0

# Example of secondary_analysis_metadata.json

metadata_example = {
    'epic_type': ['analyses', 'annotations'],
    'modalities': [{'annotations': ['azimuth_label', 'leiden'],
                    'n_obs': 22094,
                    'n_vars': 29078,
                    'name': 'HT_processed',
                    'obs_keys': ['age', 'azimuth_id', 'azimuth_label', ...],
                    'obsm_keys': ['X_pca', 'X_umap', 'annotation', 'azimuth_label', 'leiden'],
                    'var_keys': ['hugo_symbol', 'mean', 'n_cells', 'std']
                    }
                   ],
    'n_obs': 22094,
    'n_vars': 29078,
    'obs_keys': [
        # repeats the obs keys from modalities, prefixed with the modality name
        'HT_processed:age', 'HT_processed:azimuth_id', 'HT_processed:azimuth_label'
        # ... etc etc
    ],
    'obsm_keys': ['HT_processed'],
    'shape': [22094, 29078],
    'var_keys': ['hugo_symbol', 'mean', 'n_cells', 'std']
}

"""


class ObjectByAnalyteConfBuilder(ViewConfBuilder):
    def __init__(self, entity: dict, groups_token: str, assets_endpoint: str, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._scatterplot_mappings: list[str] = []

    @cached_property
    def _secondary_analysis_metadata(self):
        """Get the secondary analysis metadata json file from the entity files.
        :rtype: dict The secondary analysis metadata json file
        """
        files = self._get_file_paths()
        for file in files:
            if file.endswith("secondary_analysis_metadata.json"):
                url = super()._build_assets_url(file)
                import requests

                resp = requests.get(url)
                resp.raise_for_status()
                json = resp.json()
                if json:
                    return json
        raise ValueError(f"No secondary analysis metadata json file found for entity {self._uuid}")

    @cached_property
    def _zarr_path(self):
        """Get the zarr path from the entity files.
        :rtype: str The zarr path
        """
        files = self._get_file_paths()
        for file in files:
            if file.endswith(".zarr.zip"):
                result = super()._build_assets_url(file, use_token=True)
                # If the result is null, still raise an error
                if result:
                    return result
        raise ValueError(f"No zarr file found for entity {self._uuid}")

    @cached_property
    def zarr_store(self):  # pragma: no cover
        request_init = self._get_request_init() or {}
        return read_zip_zarr(self._zarr_path, request_init)

    @cached_property
    def _get_modalities(self):
        """
        Retrieves the modalities from the secondary analysis metadata.
        """
        return self._secondary_analysis_metadata.get("modalities", [])

    @cached_property
    def _get_epic_type(self):  # pragma: no cover
        return self._secondary_analysis_metadata.get("epic_type", [])

    def _get_obs_set_keys(self, modality):
        return modality.get("annotations", [])

    def _get_obs_set_paths(self, modality):
        """
        Get the paths to the observation sets for a given modality.
        """
        return [
            f"mod/{modality.get('name')}/obsm/annotation/{annotation}"
            for annotation in self._get_obs_set_keys(modality)
        ]

    def _get_obs_set_names(self, modality):
        """
        Get the normalized human-readable names of the annotated cell sets for a given modality.
        """
        return [annotation.replace("_", " ").title() for annotation in self._get_obs_set_keys(modality)]

    def _get_obs_embeddings(self, modality):
        non_embedding_keys = ["annotation"]
        non_embedding_keys.extend(self._get_obs_set_keys(modality))
        return [key for key in modality.get("obsm_keys", []) if key not in non_embedding_keys]

    def _get_obs_embedding_paths(self, modality):
        """
        Gets the keys in `obsm` except for `annotation` and the obs set paths
        """
        non_embedding_keys = ["annotation"]
        non_embedding_keys.extend(self._get_obs_set_keys(modality))
        return [f"mod/{modality.get('name')}/obsm/{key}" for key in self._get_obs_embeddings(modality)]

    def _get_obs_embedding_names(self, modality):
        """
        Gets the human-readable normalized names of the obs embeddings for a given modality.

        Example: ["X_umap", "X_pca"] -> ["UMAP", "PCA"]
        """
        embeddings = self._get_obs_embeddings(modality)

        formatted_embeddings = [embedding.split("_")[-1].upper() for embedding in embeddings]

        for embedding in formatted_embeddings:
            if embedding not in self._scatterplot_mappings:
                self._scatterplot_mappings.append(embedding)

        return formatted_embeddings

    def _get_feature_labels_path(self, modality):
        """
        Gets the path to the feature names (mod/{modality_name}/var/{uniprot_id | hugo_symbol})
        if it exists
        """
        var_keys = modality.get("var_keys", [])
        path_base = f"mod/{modality.get('name')}/var"
        if "hugo_symbol" in var_keys:
            return f"{path_base}/hugo_symbol"
        if "uniprot_id" in var_keys:
            return f"{path_base}/uniprot_id"
        return None

    def _get_feature_filters_path(self, modality):
        """
        Provides the path indicating the highly variable features to include in the heatmap
        """
        return f"mod/{modality.get('name')}/var/highly_variable"

    def _get_feature_matrix_path(self, modality):
        """
        Gets the path to the "X" feature matrix for the modality if it exists
        and has non-zero dimensions
        """
        if modality.get("n_obs") > 0 and modality.get("n_vars") > 0:
            return f"mod/{modality.get('name')}/X"
        return None

    def _get_obs_labels_path(self, modality):  # pragma: no cover
        """
        Gets the non-annotation obs columns
        """
        annotation_keys = self._get_obs_set_keys(modality) + ["annotation"]
        return [
            f"mod/{modality.get('name')}/obs/{key}"
            for key in modality.get("obs_keys", [])
            if key not in annotation_keys
        ]

    def _is_spatial(self, modality):
        """
        Returns whether the `obsm/X_spatial` key exists in the given modality
        """
        return "X_spatial" in modality.get("obsm_keys", [])

    def _get_spatial(self, modality):
        """
        Returns the path to the spatial coordinates for the modality if present
        """
        if self._is_spatial(modality):
            return f"mod/{modality.get('name')}/obsm/X_spatial"
        return None

    def _get_anndata_wrappers(self):
        """
        Create AnnData wrappers for each modality in the mudata object.
        """
        wrappers = []
        for modality in self._get_modalities:
            wrapper = AnnDataWrapper(
                adata_url=self._zarr_path,
                is_zip=True,
                obs_feature_matrix_path=self._get_feature_matrix_path(modality),
                obs_set_paths=self._get_obs_set_paths(modality),
                obs_set_names=self._get_obs_set_names(modality),
                obs_embedding_paths=self._get_obs_embedding_paths(modality),
                obs_embedding_names=self._get_obs_embedding_names(modality),
                obs_locations_path=self._get_spatial(modality),
                obs_embedding_dims=[[0, 1]],
                feature_labels_path=self._get_feature_labels_path(modality),
                initial_feature_filter_path=self._get_feature_filters_path(modality),
                request_init=self._get_request_init(),
            )
            wrappers.append(wrapper)
        return wrappers

    def _setup_anndata_view_config(self, vc, dataset):
        scatterplot_mappings = self._scatterplot_mappings

        scatterplots = []
        num_mappings = len(scatterplot_mappings)

        # Scatterplots take up the top left 4x6 area; the initial size of the views depends
        # on how many scatterplot embeddings are available.

        # Small helper function to cut down on repetitive boilerplate
        def add_scatterplot(mapping, x, y, w, h):
            return vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=mapping, x=x, y=y, w=w, h=h)

        if num_mappings == 1:
            scatterplot = add_scatterplot(mapping=scatterplot_mappings[0], x=0, y=0, w=4, h=6)
            scatterplots.append(scatterplot)
        elif num_mappings == 2:
            for i, mapping in enumerate(scatterplot_mappings):
                scatterplots.append(add_scatterplot(mapping=mapping, x=0, y=i * 3, w=4, h=3))
        elif num_mappings == 3:
            for i, mapping in enumerate(scatterplot_mappings):
                scatterplots.append(add_scatterplot(mapping=mapping, x=0, y=i * 2, w=4, h=2))
        elif num_mappings >= 4:
            # Currently supporting up to four scatterplots.
            available_mappings_for_visualization = scatterplot_mappings[:4]
            for i, mapping in enumerate(available_mappings_for_visualization):
                row = i // 2
                col = i % 2
                scatterplots.append(add_scatterplot(mapping=mapping, x=col * 2, y=row * 3, w=2, h=3))
        # Check if any modality has spatial data
        has_spatial = any(self._is_spatial(modality) for modality in self._get_modalities)

        spatial_view = None
        spatial_controller = None

        cell_sets_and_gene_list_w = 4 if has_spatial else 8
        cell_sets_and_gene_list_x = 8 if has_spatial else 4

        if has_spatial:
            # Add spatial views if spatial data is available
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=4, h=3)
            spatial_controller = vc.add_view("layerControllerBeta", dataset=dataset, x=4, y=3, w=4, h=3)

        cell_sets = vc.add_view(
            cm.OBS_SETS, dataset=dataset, x=cell_sets_and_gene_list_x, y=0, w=cell_sets_and_gene_list_w, h=3
        )
        gene_list = vc.add_view(
            cm.FEATURE_LIST, dataset=dataset, x=cell_sets_and_gene_list_x, y=3, w=cell_sets_and_gene_list_w, h=3
        )

        cell_sets_expr = vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset, x=7, y=6, w=5, h=4)
        heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, x=0, y=6, w=7, h=4)

        views = list(
            filter(
                lambda v: v is not None,
                [cell_sets, gene_list, *scatterplots, spatial_view, spatial_controller, cell_sets_expr, heatmap],
            )
        )

        self._views = views

        return vc

    def get_conf_cells(self, **kwargs):
        # Ensure the zarr store is present
        self._zarr_path  # noqa: B018

        vc = VitessceConfig(name=self._uuid, schema_version=self._schema_version)

        ds = vc.add_dataset(name=self._uuid)

        [ds.add_object(wrapper) for wrapper in self._get_anndata_wrappers()]

        vc = self._setup_anndata_view_config(vc, ds)

        return get_conf_cells(vc.to_dict())
