from functools import cached_property

from vitessce import AnnDataWrapper
from vitessce import Component as cm

from ..constants import MAX_OBS_FOR_HEATMAP, MAX_OBS_FOR_SPATIAL_VIEWS
from ..utils import get_conf_cells, read_zip_zarr, with_config_builder_user_agent
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


# obsm keys that hold spatial coordinates rather than a derived embedding. When one is present it
# becomes `obsLocations`, which the spatialBeta view renders, so a scatterplot over the same key
# would be a second view of the identical array. `X_spatial` is HuBMAP's key and `spatial` is the
# scanpy/squidpy convention; datasets carry both.
SPATIAL_OBSM_KEYS = ("X_spatial", "spatial")


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

                resp = requests.get(url, **with_config_builder_user_agent(self._get_request_init()))
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

    @cached_property
    def n_obs(self):
        """The largest number of observations across the mudata modalities.

        Unlike the AnnData and SPRM builders, which read the count out of the zarr store, the
        count is already in the secondary analysis metadata that this builder fetches to derive
        the layout -- so gating on it costs nothing extra. A missing count reads as 0, which
        leaves the heatmap in place.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': [{'rel_path': 'x/secondary_analysis.zarr.zip'}]}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        >>> builder.__dict__['_secondary_analysis_metadata'] = {
        ...     'n_obs': 1000, 'modalities': [{'name': 'rna', 'n_obs': 150_000}, {'name': 'atac'}]}
        >>> builder.n_obs
        150000
        """
        counts = [modality.get("n_obs") or 0 for modality in self._get_modalities]
        counts.append(self._secondary_analysis_metadata.get("n_obs") or 0)
        return max(counts)

    def _should_include_optional_views(self, view_type=None):
        """Whether an optional view should be added, following the same rules as the other builders:
        minimal configs drop every optional view, and heatmaps are also dropped for datasets too
        large to render one performantly -- the heatmap's loader densifies the whole feature matrix.
        Only the heatmap does that; the expression distribution reads one feature at a time.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': [{'rel_path': 'x/secondary_analysis.zarr.zip'}]}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        >>> builder.__dict__['n_obs'] = MAX_OBS_FOR_HEATMAP
        >>> builder._should_include_optional_views('heatmap')
        True
        >>> builder.__dict__['n_obs'] = MAX_OBS_FOR_HEATMAP + 1
        >>> builder._should_include_optional_views('heatmap')
        False
        >>> builder._should_include_optional_views('gene_list')
        True
        >>> builder._minimal = True
        >>> builder._should_include_optional_views('gene_list')
        False
        """
        if self._minimal:
            return False
        return not (view_type == "heatmap" and self.n_obs > MAX_OBS_FOR_HEATMAP)

    @cached_property
    def _include_spatial_views(self):
        """Whether to add the spatialBeta / layerControllerBeta pair.

        Requires spatial coordinates and a dataset small enough for the spot layer's
        per-observation buffers (see ``MAX_OBS_FOR_SPATIAL_VIEWS``). Over that limit the pair is
        dropped and the coordinates are shown as a scatterplot instead, which renders the same
        points far more cheaply -- so nothing is lost but the image layers and their controls.

        Unlike the other optional views this ignores ``_minimal``: minimal configs have always
        included the spatial view when the data has it, and the marker-gene preview that requests
        minimal relies on it.

        >>> def build(obsm_keys, n_obs):
        ...     entity = {'uuid': 'test', 'status': 'Published', 'files': []}
        ...     builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        ...     builder.__dict__['_secondary_analysis_metadata'] = {
        ...         'modalities': [{'name': 'rna', 'obsm_keys': obsm_keys}]}
        ...     builder.__dict__['n_obs'] = n_obs
        ...     return builder

        >>> build(['X_umap', 'X_spatial'], MAX_OBS_FOR_SPATIAL_VIEWS)._include_spatial_views
        True

        One observation too many for the spot layer:

        >>> build(['X_umap', 'X_spatial'], MAX_OBS_FOR_SPATIAL_VIEWS + 1)._include_spatial_views
        False

        No spatial coordinates at all, at any size:

        >>> build(['X_umap'], 1000)._include_spatial_views
        False
        """
        if not any(self._is_spatial(modality) for modality in self._get_modalities):
            return False
        return self.n_obs <= MAX_OBS_FOR_SPATIAL_VIEWS

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

    def _get_obs_embedding_pairs(self, modality):
        """Ordered ``(obsm_key, display_name)`` pairs for one modality's scatterplot embeddings.

        Three kinds of key in ``obsm_keys`` are not scatterplot embeddings:

        - ``annotation`` and the annotated cell set keys, which are obs sets.
        - The spatial coordinate keys, when the modality has them *and* the spatial view is being
          added. They become ``obsLocations`` and spatialBeta draws them, so a ``SPATIAL``
          scatterplot would be a second view of the same array. When the dataset is too large for
          the spatial view, the keys are left in: the scatterplot then becomes the only thing
          rendering those coordinates.
        - A key whose display name is already taken. A name is the key's last
          underscore-delimited segment, so distinct keys can collapse onto one -- Vitessce
          addresses an embedding by that name, making the second unreachable, a fetch of a whole
          array that buys nothing. First key wins.

        Dedupe is per modality on purpose: two modalities legitimately both expose ``UMAP``, and
        one wrapper per modality sharing an ``embeddingType`` is how Vitessce coordinates them.

        >>> def build(include_spatial_views):
        ...     entity = {'uuid': 'test', 'status': 'Published', 'files': []}
        ...     builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        ...     builder.__dict__['_include_spatial_views'] = include_spatial_views
        ...     return builder
        >>> modality = {'name': 'rna', 'annotations': ['leiden'],
        ...             'obsm_keys': ['X_pca', 'X_spatial', 'X_spatial_gpr', 'X_umap',
        ...                           'spatial', 'annotation', 'leiden']}

        With the spatial view present, both spatial keys are left to ``obsLocations``, while the
        GPR-smoothed embedding is a genuinely separate projection and stays:

        >>> build(True)._get_obs_embedding_pairs(modality)
        [('X_pca', 'PCA'), ('X_spatial_gpr', 'GPR'), ('X_umap', 'UMAP')]

        Without it -- too many observations for the spot layer -- ``X_spatial`` comes back as a
        scatterplot, which is then the only view of those coordinates:

        >>> build(False)._get_obs_embedding_pairs(modality)
        [('X_pca', 'PCA'), ('X_spatial', 'SPATIAL'), ('X_spatial_gpr', 'GPR'), ('X_umap', 'UMAP')]

        A modality with no spatial coordinates never consults the gate, and names still dedupe:

        >>> build(True)._get_obs_embedding_pairs({'name': 'rna', 'obsm_keys': ['X_umap', 'umap', 'X_pca']})
        [('X_umap', 'UMAP'), ('X_pca', 'PCA')]
        """
        non_embedding_keys = ["annotation", *self._get_obs_set_keys(modality)]
        if self._is_spatial(modality) and self._include_spatial_views:
            non_embedding_keys.extend(SPATIAL_OBSM_KEYS)
        pairs = []
        seen_names = set()
        for key in modality.get("obsm_keys", []):
            if key in non_embedding_keys:
                continue
            name = key.split("_")[-1].upper()
            if name in seen_names:
                continue
            seen_names.add(name)
            pairs.append((key, name))
        return pairs

    def _get_obs_embeddings(self, modality):
        return [key for key, _name in self._get_obs_embedding_pairs(modality)]

    def _get_obs_embedding_paths(self, modality):
        """
        Gets the keys in `obsm` except for `annotation` and the obs set paths
        """
        return [f"mod/{modality.get('name')}/obsm/{key}" for key in self._get_obs_embeddings(modality)]

    def _get_obs_embedding_names(self, modality):
        """
        Gets the human-readable normalized names of the obs embeddings for a given modality.

        Example: ["X_umap", "X_pca"] -> ["UMAP", "PCA"]
        """
        formatted_embeddings = [name for _key, name in self._get_obs_embedding_pairs(modality)]

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
        if (modality.get("n_obs") or 0) > 0 and (modality.get("n_vars") or 0) > 0:
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
                # No obs_embedding_dims: vitessce already defaults every obsEmbedding entry to
                # [0, 1], and the argument only overrides entries positionally -- so [[0, 1]] set
                # index 0 to the value it already had, and raised IndexError for a modality whose
                # only obsm keys are spatial (hence no embeddings) or annotations.
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
        # Spatial data present, and few enough observations for the spot layer to handle.
        has_spatial = self._include_spatial_views

        spatial_view = None
        spatial_controller = None

        cell_sets_and_gene_list_w = 4 if has_spatial else 8
        cell_sets_and_gene_list_x = 8 if has_spatial else 4

        if has_spatial:
            # Add spatial views if spatial data is available
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=4, h=3)
            spatial_controller = vc.add_view("layerControllerBeta", dataset=dataset, x=4, y=3, w=4, h=3)

        include_gene_list = self._should_include_optional_views("gene_list")
        include_heatmap = self._should_include_optional_views("heatmap")

        # Without the gene list, the cell sets take over the whole right column.
        cell_sets = vc.add_view(
            cm.OBS_SETS,
            dataset=dataset,
            x=cell_sets_and_gene_list_x,
            y=0,
            w=cell_sets_and_gene_list_w,
            h=3 if include_gene_list else 6,
        )
        gene_list = None
        if include_gene_list:
            gene_list = vc.add_view(
                cm.FEATURE_LIST, dataset=dataset, x=cell_sets_and_gene_list_x, y=3, w=cell_sets_and_gene_list_w, h=3
            )

        # Without the heatmap, the expression distribution spans the whole bottom row.
        cell_sets_expr = vc.add_view(
            cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION,
            dataset=dataset,
            x=7 if include_heatmap else 0,
            y=6,
            w=5 if include_heatmap else 12,
            h=4,
        )
        heatmap = None
        if include_heatmap:
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

        vc, ds = self._create_vitessce_config()

        [ds.add_object(wrapper) for wrapper in self._get_anndata_wrappers()]

        vc = self._setup_anndata_view_config(vc, ds)

        return get_conf_cells(vc)
