"""
Data loading utilities for sinkhole modeling
"""
import os
import pandas as pd
import geopandas as gpd
import shapely.wkb
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Optional, Set, Tuple
from sklearn.cluster import KMeans
import torch
from torch_geometric.data import Data
from src.models.sinkhole_modeling.config import log, DB_ENV_VAR


def load_dataset(table: str = "feature_matrix_25_geo",
                 chunksize: int = 100_000) -> gpd.GeoDataFrame:
    """Stream the entire table, ensure geometry, return GeoDataFrame."""
    load_dotenv()
    dsn = os.getenv(DB_ENV_VAR)
    if not dsn:
        raise RuntimeError(f"Environment variable {DB_ENV_VAR} is not set.")
    engine = create_engine(dsn)

    gdf_parts: list[pd.DataFrame] = []
    sql = f"SELECT * FROM {table}"
    for chunk in pd.read_sql_query(sql, engine, chunksize=chunksize):
        if "geom" not in chunk.columns:
            raise ValueError("Column 'geom' missing – confirm PostGIS WKB hex present.")
        chunk["geometry"] = chunk["geom"].apply(lambda h: shapely.wkb.loads(bytes.fromhex(h)))
        chunk = chunk.drop(columns=["geom"])
        gdf_parts.append(chunk)
    gdf = gpd.GeoDataFrame(pd.concat(gdf_parts, ignore_index=True), geometry="geometry")
    if "subsidence_occurrence" not in gdf.columns:
        raise ValueError("Required label column 'subsidence_occurrence' missing.")
    pos = int(gdf["subsidence_occurrence"].sum())
    log(f"Loaded {len(gdf):,} rows  •  positives: {pos} ({pos / len(gdf):.4%})", level=2)
    return gdf


def get_silent_grid_ids(X: pd.DataFrame, y: pd.Series, percentile: int = 90) -> Set[int]:
    """
    Dynamically generate silent grid IDs based on low scoring areas

    Args:
        X: Feature dataframe that must include 'grid_id' column
        y: Target variable
        percentile: Percentile cutoff for identifying silent zones

    Returns:
        Set of grid IDs identified as silent zones
    """
    if 'grid_id' not in X.columns:
        raise ValueError("X must contain 'grid_id' column to generate silent grid IDs")

    # Get indices where subsidence hasn't occurred
    non_subsidence_mask = (y == 0)

    # Calculate scores based on distance feature (higher is further from known sinkholes)
    if 'min_distance_to_sinkhole' in X.columns:
        # Higher distance = higher score = more likely to be silent zone
        distance_scores = X.loc[non_subsidence_mask, 'min_distance_to_sinkhole'].values

        # Take top percentile as silent zones
        threshold = np.percentile(distance_scores, percentile)
        silent_mask = (distance_scores >= threshold)

        # Get grid IDs of silent zones
        silent_grid_ids = set(X.loc[non_subsidence_mask].iloc[silent_mask]['grid_id'].tolist())

        log(f"Dynamically generated {len(silent_grid_ids)} silent grid IDs (percentile {percentile})", level=1)

        return silent_grid_ids
    else:
        # Fallback if distance feature not available: randomly select 5% of non-subsidence areas
        non_subsidence_ids = X.loc[non_subsidence_mask, 'grid_id'].values
        n_silent = max(int(len(non_subsidence_ids) * 0.05), 10)
        silent_grid_ids = set(np.random.choice(non_subsidence_ids, size=n_silent, replace=False))

        log(f"Generated {len(silent_grid_ids)} random silent grid IDs (no distance feature available)", level=1)
        return silent_grid_ids


def create_spatial_blocks(gdf: gpd.GeoDataFrame, n_blocks: int = 5) -> np.ndarray:
    """
    Create spatial blocks for cross-validation based on coordinates
    Returns array of block indices for each row in gdf
    """
    # Extract centroids if geometry column exists
    if 'geometry' in gdf.columns:
        X = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    else:
        raise ValueError("GeoDataFrame must have 'geometry' column")

    # Apply KMeans clustering to create spatial blocks
    kmeans = KMeans(n_clusters=n_blocks, random_state=42)
    blocks = kmeans.fit_predict(X)

    log(f"Created {n_blocks} spatial blocks with sizes: {pd.Series(blocks).value_counts().to_dict()}")

    return blocks


def build_knn_graph(gdf: gpd.GeoDataFrame, k: int = 8, max_distance: float = 2000) -> np.ndarray:
    """
    Build k-nearest neighbors graph from GeoDataFrame

    Args:
        gdf: GeoDataFrame with geometry column
        k: Number of neighbors
        max_distance: Maximum distance in meters

    Returns:
        Array of shape [2, num_edges] containing edge indices
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # Extract centroids
    if 'geometry' not in gdf.columns:
        raise ValueError("GeoDataFrame must contain 'geometry' column")

    centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry])

    # Build KNN using sklearn
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)

    # Generate edges
    edge_list = []
    for i in range(len(gdf)):
        for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self-loop
            if dist <= max_distance:
                edge_list.append((i, j))

    # Convert to PyTorch format
    edges = np.array(edge_list).T

    log(f"Built KNN graph with {len(edge_list)} edges for {len(gdf)} nodes", level=2)

    return edges


def prepare_pyg_data(gdf: gpd.GeoDataFrame, feature_cols: list, label_col: str = 'subsidence_occurrence',
                     k_neighbors: int = 8, max_distance: float = 2000) -> Data:
    """
    Prepare PyTorch Geometric Data object from GeoDataFrame

    Args:
        gdf: GeoDataFrame with data
        feature_cols: List of feature column names
        label_col: Label column name
        k_neighbors: Number of neighbors for graph construction
        max_distance: Maximum distance in meters for graph construction

    Returns:
        PyTorch Geometric Data object
    """
    import torch
    from torch_geometric.data import Data

    # Extract features and labels
    X = gdf[feature_cols].values
    y = gdf[label_col].values

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Build graph
    edge_index = build_knn_graph(gdf, k=k_neighbors, max_distance=max_distance)

    # Create PyG data object
    data = Data(
        x=torch.FloatTensor(X),
        y=torch.LongTensor(y),
        edge_index=torch.LongTensor(edge_index)
    )

    log(f"Prepared PyG data with {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features", level=1)

    return data


def add_spatial_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add spatial features to the GeoDataFrame for improved graph learning

    Args:
        gdf: GeoDataFrame with geometry column

    Returns:
        GeoDataFrame with added spatial features
    """
    # Extract centroids
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y

    # Calculate area and perimeter
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length

    # Calculate compactness (circularity)
    gdf['compactness'] = 4 * np.pi * gdf['area'] / (gdf['perimeter'] ** 2)

    # Scale features
    for col in ['centroid_x', 'centroid_y', 'area', 'perimeter', 'compactness']:
        gdf[col] = (gdf[col] - gdf[col].mean()) / (gdf[col].std() + 1e-8)

    log(f"Added spatial features to GeoDataFrame: {['centroid_x', 'centroid_y', 'area', 'perimeter', 'compactness']}",
        level=2)

    return gdf