from functools import cached_property
from math import ceil

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

# obsm keys that are never useful as a scatterplot embedding. `X_spatial_gpr` is a Gaussian-process
# regression fitted over the spatial coordinates rather than a projection of the observations, and
# renders as nothing meaningful. Listed explicitly rather than matched by pattern, so that adding
# one stays a deliberate decision about a key someone has actually looked at.
NON_EMBEDDING_OBSM_KEYS = ("X_spatial_gpr",)

# Most scatterplots to lay out, however many embeddings a modality exposes.
MAX_SCATTERPLOTS = 4
# Views dropped above MAX_OBS_FOR_HEATMAP. Only the heatmap qualifies: its loader densifies the
# whole feature matrix, and no dataset-side option avoids that. The expression-by-cell-set
# distribution reads only the selected feature's column plus obs-set membership, so it was gated
# alongside the heatmap only until vitessce made both of those bounded; it now stays at any size.
EXPRESSION_SUMMARY_VIEWS = ("heatmap",)


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
        minimal configs drop every optional view, and the views that need more of the feature
        matrix than a per-feature slice (``EXPRESSION_SUMMARY_VIEWS``) are dropped for datasets
        too large to build that summary in the browser.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': [{'rel_path': 'x/secondary_analysis.zarr.zip'}]}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        >>> builder.__dict__['n_obs'] = MAX_OBS_FOR_HEATMAP
        >>> builder._should_include_optional_views('heatmap')
        True
        >>> builder.__dict__['n_obs'] = MAX_OBS_FOR_HEATMAP + 1
        >>> builder._should_include_optional_views('heatmap')
        False

        The gene list only reads the feature index, and the expression distribution only the
        selected feature's column, so size never affects either:

        >>> builder._should_include_optional_views('gene_list')
        True
        >>> builder._should_include_optional_views('expression_distribution')
        True
        >>> builder._minimal = True
        >>> builder._should_include_optional_views('gene_list')
        False
        >>> builder._should_include_optional_views('expression_distribution')
        False
        """
        if self._minimal:
            return False
        return not (view_type in EXPRESSION_SUMMARY_VIEWS and self.n_obs > MAX_OBS_FOR_HEATMAP)

    @cached_property
    def _include_spatial_views(self):
        """Whether to add the spatialBeta / layerControllerBeta pair.

        Requires spatial coordinates and a dataset small enough for the spot layer's
        per-observation buffers (see ``MAX_OBS_FOR_SPATIAL_VIEWS``). Over that limit the pair is
        dropped and the spatial coordinates are not rendered at all: a scatterplot over them was
        still enough to keep the tab from loading at ~2M observations. ``obsLocations`` is then
        left out of the wrapper too, since nothing remains to read it.

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
        """Paths to the annotated cell sets for a modality.

        These labels live twice in the store: as ``obs/<annotation>`` and as a column of the
        ``obsm/annotation`` dataframe. Prefer ``obs``: the source caches the obs index per obs
        path, so reading the sets from there reuses the index the rest of the config already
        loaded, where ``obsm/annotation`` makes it download and materialize a second string array
        with one entry per observation. On a multi-million-observation dataset that second copy is
        tens of MB transferred and hundreds of MB of heap for nothing.

        Falls back to ``obsm/annotation`` for an annotation that isn't in ``obs_keys``.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': []}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        >>> builder._get_obs_set_paths({'name': 'rna', 'annotations': ['azimuth_label', 'leiden'],
        ...                            'obs_keys': ['age', 'azimuth_label', 'leiden']})
        ['mod/rna/obs/azimuth_label', 'mod/rna/obs/leiden']

        >>> builder._get_obs_set_paths({'name': 'rna', 'annotations': ['azimuth_label'],
        ...                            'obs_keys': ['age']})
        ['mod/rna/obsm/annotation/azimuth_label']
        """
        obs_keys = modality.get("obs_keys", [])
        base = f"mod/{modality.get('name')}"
        return [
            f"{base}/obs/{annotation}" if annotation in obs_keys else f"{base}/obsm/annotation/{annotation}"
            for annotation in self._get_obs_set_keys(modality)
        ]

    def _get_obs_set_names(self, modality):
        """
        Get the normalized human-readable names of the annotated cell sets for a given modality.
        """
        return [annotation.replace("_", " ").title() for annotation in self._get_obs_set_keys(modality)]

    def _get_obs_embedding_pairs(self, modality):
        """Ordered ``(obsm_key, display_name)`` pairs for one modality's scatterplot embeddings.

        Four kinds of key in ``obsm_keys`` are not scatterplot embeddings:

        - ``annotation`` and the annotated cell set keys, which are obs sets.
        - Anything in ``NON_EMBEDDING_OBSM_KEYS``, which is not a projection of the observations.
        - The spatial coordinate keys. Where the spatial view is added they are its
          ``obsLocations``, so a ``SPATIAL`` scatterplot would be a second view of the same array;
          where it is not, the dataset was too large for the spatial coordinates to render at all.
          Either way they are not a scatterplot.
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

        The spatial keys and the GPR fit both drop out, whether or not the spatial view is added:

        >>> build(True)._get_obs_embedding_pairs(modality)
        [('X_pca', 'PCA'), ('X_umap', 'UMAP')]
        >>> build(False)._get_obs_embedding_pairs(modality)
        [('X_pca', 'PCA'), ('X_umap', 'UMAP')]

        Names still dedupe -- ``X_umap`` and ``umap`` both read as ``UMAP``, first key wins:

        >>> build(True)._get_obs_embedding_pairs({'name': 'rna', 'obsm_keys': ['X_umap', 'umap', 'X_pca']})
        [('X_umap', 'UMAP'), ('X_pca', 'PCA')]
        """
        non_embedding_keys = [
            "annotation",
            *NON_EMBEDDING_OBSM_KEYS,
            *SPATIAL_OBSM_KEYS,
            *self._get_obs_set_keys(modality),
        ]
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
        """Path to the boolean column restricting which features the heatmap loads.

        Only claimed when the modality actually has the column. Emitting it unconditionally
        pointed at a nonexistent array for every modality without ``highly_variable`` -- inert
        while no view reads the feature matrix, but a landmine for one that does.

        Where the column exists it is doing real work: it is what keeps the heatmap's loader from
        densifying all of ``X`` (see ``EXPRESSION_SUMMARY_VIEWS``), so it is kept rather than
        dropped outright.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': []}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')
        >>> builder._get_feature_filters_path({'name': 'rna', 'var_keys': ['highly_variable']})
        'mod/rna/var/highly_variable'
        >>> repr(builder._get_feature_filters_path({'name': 'rna', 'var_keys': ['hugo_symbol']}))
        'None'
        """
        if "highly_variable" not in modality.get("var_keys", []):
            return None
        return f"mod/{modality.get('name')}/var/highly_variable"

    def _get_feature_matrix_path(self, modality):
        """Path to the expression matrix for a modality with non-zero dimensions.

        Prefers ``layers/unscaled`` over ``X`` where the modality carries it: the unscaled values
        read as raw expression rather than z-scores, which is what a feature color scale wants.
        Since object-by-analyte-to-ui 0.0.7 both are written CSC, so a gene selection slices
        ``[indptr[i], indptr[i + 1])`` either way.

        ``layers`` is listed in ``secondary_analysis_metadata.json`` only from that same container
        version. Where it is present it is authoritative. Without it there is no way to know which
        layers exist, so fall back to the older inference: ``mean`` and ``std`` are what a scaling
        step writes to ``var_keys``, and are taken as the signal that ``X`` was scaled and the
        unscaled values kept alongside it. Layer names come from the submitter rather than the
        container, so that inference can name a layer the store does not have -- it survives only
        for already-processed datasets, and goes away as they are reprocessed.

        >>> entity = {'uuid': 'test', 'status': 'Published', 'files': []}
        >>> builder = ObjectByAnalyteConfBuilder(entity, 'token', 'https://example.com')

        ``layers`` decides it where present:

        >>> builder._get_feature_matrix_path({'name': 'rna', 'n_obs': 10, 'n_vars': 5,
        ...                                   'layers': ['raw', 'unscaled']})
        'mod/rna/layers/unscaled'

        including where it contradicts the ``mean``/``std`` inference -- the case that used to
        yield a path to an array the container never wrote:

        >>> builder._get_feature_matrix_path({'name': 'rna', 'n_obs': 10, 'n_vars': 5,
        ...                                   'layers': ['normalized', 'raw'],
        ...                                   'var_keys': ['hugo_symbol', 'mean', 'std']})
        'mod/rna/X'

        Metadata written before 0.0.7 has no ``layers`` key, and keeps the inference:

        >>> builder._get_feature_matrix_path({'name': 'rna', 'n_obs': 10, 'n_vars': 5,
        ...                                   'var_keys': ['hugo_symbol', 'mean', 'std']})
        'mod/rna/layers/unscaled'

        >>> builder._get_feature_matrix_path({'name': 'rna', 'n_obs': 10, 'n_vars': 5,
        ...                                   'var_keys': ['hugo_symbol']})
        'mod/rna/X'

        >>> repr(builder._get_feature_matrix_path({'name': 'rna', 'n_obs': 0, 'n_vars': 5}))
        'None'
        """
        if (modality.get("n_obs") or 0) > 0 and (modality.get("n_vars") or 0) > 0:
            base = f"mod/{modality.get('name')}"
            layers = modality.get("layers")
            if layers is not None:
                return f"{base}/layers/unscaled" if "unscaled" in layers else f"{base}/X"
            var_keys = modality.get("var_keys", [])
            if "mean" in var_keys and "std" in var_keys:
                return f"{base}/layers/unscaled"
            return f"{base}/X"
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
        Returns the path to the spatial coordinates for the modality, if present and if a view
        will actually render them.
        """
        if self._is_spatial(modality) and self._include_spatial_views:
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

        # Spatial data present, and few enough observations for the spot layer to handle.
        has_spatial = self._include_spatial_views

        # The right-hand column is always 4 wide. The scatterplots get everything to its left, so
        # when there is no spatial pair the plots widen into that space instead of the cell sets
        # and gene list stretching across it.
        cell_sets_and_gene_list_w = 4
        cell_sets_and_gene_list_x = 8
        scatterplots_w = 4 if has_spatial else 8

        # Small helper function to cut down on repetitive boilerplate
        def add_scatterplot(mapping, x, y, w, h):
            return vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=mapping, x=x, y=y, w=w, h=h)

        # Fill the scatterplots_w x 6 region row-major: a single column while three or fewer fit
        # legibly, two columns beyond that. A modality whose only obsm keys are spatial or
        # annotations has no embedding to plot at all.
        mappings = scatterplot_mappings[:MAX_SCATTERPLOTS]
        scatterplots = []
        if mappings:
            cols = 1 if len(mappings) <= 3 else 2
            rows = ceil(len(mappings) / cols)
            cell_w = scatterplots_w // cols
            cell_h = 6 // rows
            scatterplots = [
                add_scatterplot(
                    mapping=mapping,
                    x=(i % cols) * cell_w,
                    y=(i // cols) * cell_h,
                    w=cell_w,
                    h=cell_h,
                )
                for i, mapping in enumerate(mappings)
            ]

        spatial_view = None
        spatial_controller = None

        if has_spatial:
            # Add spatial views if spatial data is available
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=4, h=3)
            spatial_controller = vc.add_view("layerControllerBeta", dataset=dataset, x=4, y=3, w=4, h=3)

        include_gene_list = self._should_include_optional_views("gene_list")
        include_heatmap = self._should_include_optional_views("heatmap")
        include_expression_distribution = self._should_include_optional_views("expression_distribution")

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

        # The heatmap is the only bottom-row view large datasets lose (see
        # EXPRESSION_SUMMARY_VIEWS), and without it the distribution takes the whole row. When
        # both are gone so is the row; the grid is relative, so the views above grow to fill the
        # space rather than leaving a gap.
        heatmap = None
        cell_sets_expr = None
        if include_heatmap:
            heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, x=0, y=6, w=7, h=4)
        if include_expression_distribution:
            cell_sets_expr = vc.add_view(
                cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION,
                dataset=dataset,
                x=7 if include_heatmap else 0,
                y=6,
                w=5 if include_heatmap else 12,
                h=4,
            )

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
