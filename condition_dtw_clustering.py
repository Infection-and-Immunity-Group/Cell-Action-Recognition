import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from tslearn.metrics import dtw
import umap

def plot_tracks_from_origin(file_paths, max_tracks_per_file=10, cluster_df=None):
    """
    Plot tracks from their origin point (translating all tracks to start at 0,0)
    
    Parameters:
    file_paths (list): List of paths to Excel files with tracking data
    max_tracks_per_file (int): Maximum number of tracks to plot per file to avoid overcrowding
    cluster_df (DataFrame): Optional DataFrame containing cluster assignments for tracks
    
    Returns:
    None: Plots are saved as PNG files
    """
    plt.figure(figsize=(12, 10))
    
    # Create a colormap for files or clusters
    if cluster_df is not None:
        # Use cluster colors
        unique_clusters = sorted(cluster_df['Cluster'].unique())
        n_colors = len(unique_clusters)
        colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
        color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
        plot_by = 'cluster'
    else:
        # Use file colors
        n_colors = len(file_paths)
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        color_map = {os.path.basename(file_path): colors[i] for i, file_path in enumerate(file_paths)}
        plot_by = 'file'
    
    # Process each file
    for file_idx, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        print(f"Processing positions from {file_name}...")
        
        try:
            # Load the Position sheet
            position_df = pd.read_excel(file_path, sheet_name='Position')
            
            
            # Check if required columns exist
            required_cols = ['Position X', 'Position Y', 'TrackID', 'Time']
            if not all(col in position_df.columns for col in required_cols):
                print(f"  Missing required columns in {file_name}, skipping...")
                continue
                
            # Group by TrackID
            track_groups = position_df.groupby('TrackID')
            
            # Sample tracks if there are too many
            track_ids = list(track_groups.groups.keys())
            # if len(track_ids) > max_tracks_per_file:
            #     track_ids = np.random.choice(track_ids, max_tracks_per_file, replace=False)
            
            # Plot each track
            for track_id in track_ids:
                track_data = track_groups.get_group(track_id).sort_values('Time')
                
                # Skip tracks with only one point
                if len(track_data) <= 1:
                    continue
                
                # Calculate positions relative to the first point (origin)
                x = track_data['Position X'].values - track_data['Position X'].values[0]
                y = track_data['Position Y'].values - track_data['Position Y'].values[0]
                
                # Determine color based on plot mode
                if plot_by == 'cluster':
                    # Find the cluster for this track
                    full_track_id = f"{file_name}_{track_id}"
                    if full_track_id in cluster_df.index:
                        cluster = cluster_df.loc[full_track_id, 'Cluster']
                        color = color_map[cluster]
                        label = f"Cluster {cluster}" if track_id == track_ids[0] and file_idx == 0 else None
                    else:
                        # If track not found in cluster data, use file color
                        color = color_map.get(0, 'gray')  # Default to first cluster or gray
                        label = None
                else:
                    color = color_map[file_name]
                    label = file_name if track_id == track_ids[0] else None
                
                # Plot the track
                plt.plot(x, y, '-', linewidth=1, alpha=0.7)    
                #plt.plot(x, y, '-', linewidth=1, alpha=0.7, color=color, label=label)
        
        except Exception as e:
            print(f"Error processing positions from {file_name}: {e}")
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Files" if plot_by == 'file' else "Clusters")
    
    # Add plot details
    plt.title('Cell Tracking Paths (Relative to Origin)')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save figure
    plt.tight_layout()
    plt.savefig('tracks_from_origin.svg', dpi=300)
    plt.show()

def plot_tracks_by_cluster(file_paths, features_df, n_clusters):
    """
    Plot tracks from each cluster separately, starting from origin
    
    Parameters:
    file_paths (list): List of paths to Excel files with tracking data
    features_df (DataFrame): DataFrame containing cluster assignments for tracks
    n_clusters (int): Number of clusters
    
    Returns:
    None: Plots are saved as PNG files
    """
    # Plot all tracks colored by cluster in one figure
    plot_tracks_from_origin(file_paths, max_tracks_per_file=10, cluster_df=features_df)
    
    # Plot each cluster separately
    for cluster in range(n_clusters):
        plt.figure(figsize=(10, 8))
        
        # Get tracks in this cluster
        cluster_tracks = features_df[features_df['Cluster'] == cluster].index.tolist()
        
        # Track count per file for this cluster
        track_count = {}
        
        # Process each file
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            print(f"Processing cluster {cluster} tracks from {file_name}...")
            
            try:
                # Load the Position sheet
                position_df = pd.read_excel(file_path, sheet_name='Position')
                
                # Check if required columns exist
                required_cols = ['Position X', 'Position Y', 'TrackID', 'Time']
                if not all(col in position_df.columns for col in required_cols):
                    print(f"  Missing required columns in {file_name}, skipping...")
                    continue
                    
                # Find tracks from this file that belong to the current cluster
                file_cluster_tracks = [
                    int(track_id.split('_')[-1]) 
                    for track_id in cluster_tracks 
                    if track_id.startswith(f"{file_name}_")
                ]
                
                track_count[file_name] = len(file_cluster_tracks)
                
                # Plot each track in this cluster
                for track_id in file_cluster_tracks:
                    if track_id not in position_df['TrackID'].values:
                        continue
                        
                    track_data = position_df[position_df['TrackID'] == track_id].sort_values('Time')
                    
                    # Skip tracks with only one point
                    if len(track_data) <= 1:
                        continue
                    
                    # Calculate positions relative to the first point (origin)
                    x = track_data['Position X'].values - track_data['Position X'].values[0]
                    y = track_data['Position Y'].values - track_data['Position Y'].values[0]
                    
                    # Plot the track
                    plt.plot(x, y, '-', linewidth=1, alpha=0.7)
            
            except Exception as e:
                print(f"Error processing positions from {file_name} for cluster {cluster}: {e}")
        
        # Add plot details
        plt.title(f'Cluster {cluster} Tracks (n={sum(track_count.values())})')
        plt.xlabel('X Position (μm)')
        plt.ylabel('Y Position (μm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')  # Equal aspect ratio
        
        # Add track counts per file as text
        y_pos = 0.02
        for file_name, count in track_count.items():
            plt.figtext(0.02, y_pos, f"{file_name}: {count} tracks", fontsize=8)
            y_pos += 0.03
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'cluster_{cluster}_tracks.svg', dpi=300)
        plt.show()

def load_data(file_path='tracking_data.xlsx'):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    # Group data by Track_ID and ensure it's sorted by Time_Point
    tracks = {}
    track_features = {}
    
    for track_id, group in df.groupby('Track_ID'):
        group_sorted = group.sort_values('Time_Point')
        # Select the features we want to use for DTW
        features = group_sorted[['Speed_um_min', 'Displacement_um', 'Directionality', 'Arrest_Coefficient']].values
        tracks[track_id] = features
        
        # Store feature averages for later visualization
        track_features[track_id] = {
            'Mean Speed': np.mean(group_sorted['Speed_um_min']),
            'Mean Displacement': np.mean(group_sorted['Displacement_um']),
            'Mean Directionality': np.mean(group_sorted['Directionality']),
            'Mean Arrest Coefficient': np.mean(group_sorted['Arrest_Coefficient']),
        }
    
    # Convert track features to DataFrame
    features_df = pd.DataFrame.from_dict(track_features, orient='index')
    
    return tracks, features_df

def compute_dtw_distances(tracks):
    track_ids = list(tracks.keys())
    n_tracks = len(track_ids)
    
    # Initialize distance matrix
    dtw_distances = np.zeros((n_tracks, n_tracks))
    
    # Compute DTW distances
    for i in range(n_tracks):
        if i % 10 == 0:
            print(f"Processing track {i+1}/{n_tracks}")
        for j in range(i, n_tracks):
            if i == j:
                dtw_distances[i, j] = 0
            else:
                # Calculate multivariate DTW distance
                distance = dtw(tracks[track_ids[i]], tracks[track_ids[j]])
                dtw_distances[i, j] = distance
                dtw_distances[j, i] = distance
    
    return dtw_distances, track_ids

def visualize_umap(dtw_distances, track_ids, features_df, n_clusters):
    # Configure UMAP
    reducer = umap.UMAP(
        n_components=2,
        metric='precomputed',  # We're using a precomputed distance matrix
        random_state=42#,
        #n_neighbors=5,
        #min_dist=0.0
    )
    
    # Apply UMAP to the distance matrix
    embedding = reducer.fit_transform(dtw_distances)#/np.max(dtw_distances))
    
    # Apply KMeans clustering on the embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)
    
    # Add cluster assignments to features dataframe
    features_df['Cluster'] = clusters
    
    # Plot the UMAP results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')
    
    # Add annotations for some points (can be disabled for large datasets)
    if len(track_ids) <= 50:  # Only annotate if there aren't too many points
        for i, track_id in enumerate(track_ids):
            plt.annotate(str(track_id), (embedding[i, 0], embedding[i, 1]), fontsize=8)
    
    plt.title('UMAP Projection of DTW Distance Matrix')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('track_clustering_umap.svg', dpi=300)
    plt.show()
    
    return features_df

def create_feature_pairplot(features_df):
    # Select columns for the pairplot (you can adjust this list)
    plot_features = [
        'Mean Speed', 
        'Mean Displacement', 
        'Mean Directionality', 
        'Mean Arrest Coefficient',
        'Cluster'
    ]
    
    # Create the pairplot with cluster-based coloring
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(
        features_df[plot_features], 
        hue='Cluster',
        palette='viridis',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 50},
        height=2.5
    )
    
    pairplot.fig.suptitle('Feature Relationships by Cluster', y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('track_features_pairplot.svg', dpi=300)
    plt.show()
    
    # Also create a correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = features_df[plot_features[:-1]].corr()  # Exclude the Cluster column
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('feature_correlation_heatmap.svg', dpi=300)
    plt.show()

def plot_radar_chart(features_df, n_clusters):
    """
    Create a radar chart to visualize the average feature values across clusters.

    Parameters:
    features_df (DataFrame): DataFrame containing feature values and cluster assignments
    n_clusters (int): Number of clusters
    """
    # Select features for the radar chart (excluding the Cluster column)
    radar_features = [
        'Mean Speed', 
        'Mean Displacement', 
        'Mean Directionality', 
        'Mean Arrest Coefficient',
    ]

    # Number of variables
    num_vars = len(radar_features)

    # Get cluster means and normalize them for the radar chart
    cluster_means = []
    for i in range(n_clusters):
        cluster_data = features_df[features_df['Cluster'] == i]
        # Calculate means for this cluster
        means = cluster_data[radar_features].mean()
        cluster_means.append(means)

    # Convert to DataFrame
    cluster_means_df = pd.DataFrame(cluster_means)

    # Normalize the data (0-1 scale for radar chart)
    normalized_means = (cluster_means_df - cluster_means_df.min()) / (cluster_means_df.max() - cluster_means_df.min())

    # Set up the angles for each feature
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    # Close the circle
    angles += angles[:1]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each cluster
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        values = normalized_means.iloc[i].values.tolist()
        # Close the circle by appending the first value
        values += values[:1]

        # Plot the cluster
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.tick_params(axis='x', which='major', pad=50)
    ax.set_xticklabels(radar_features)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # Customize the y-axis tick labels to show integers when the value is a whole number
    yticks = ax.get_yticks()
    yticklabels = [f"{y:.0f}" if y.is_integer() else f"{y:.1f}" for y in yticks]
    ax.set_yticklabels(yticklabels)
    plt.title('Radar Chart of Normalized Feature Means by Cluster', size=15)
    plt.tight_layout()

    # Save figure
    plt.savefig('cluster_radar_chart.svg', dpi=300)
    plt.show()

def process_multiple_files(file_paths, n_clusters=4):
    """
    Process multiple tracking data files, perform clustering on combined data,
    and create a heatmap of cluster frequencies.
    
    Parameters:
    file_paths (list): List of paths to Excel files with tracking data
    n_clusters (int): Number of clusters for k-means
    
    Returns:
    tuple: Combined DataFrame, features DataFrame with cluster assignments, and cluster counts by file
    """
    all_tracks = {}
    all_features = []
    file_to_track_ids = {}
    
    print(f"Processing {len(file_paths)} files...")
    
    # Step 1: Load and preprocess all files
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Loading data from {file_name}...")
        
        try:
            df = load_data(file_path)
            tracks, features_df = preprocess_data(df)
            
            # Add file identifier to track IDs to avoid conflicts
            renamed_tracks = {}
            file_tracks = []
            
            for track_id, track_data in tracks.items():
                new_track_id = f"{file_name}_{track_id}"
                renamed_tracks[new_track_id] = track_data
                file_tracks.append(new_track_id)
            
            # Store the mapping from file to track IDs
            file_to_track_ids[file_name] = file_tracks
            
            # Rename the index in features_df
            features_df.index = [f"{file_name}_{idx}" for idx in features_df.index]
            
            # Merge the tracks and features
            all_tracks.update(renamed_tracks)
            all_features.append(features_df)
            
            print(f"  Added {len(tracks)} tracks from {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Combine all features
    combined_features_df = pd.concat(all_features)
    print(f"Combined data contains {len(all_tracks)} tracks.")
    
    # Step 2: Compute DTW distances for all tracks
    print("Computing DTW distances for all tracks...")
    dtw_distances, track_ids = compute_dtw_distances(all_tracks)
    
    # Step 3: Apply UMAP and clustering
    print(f"Applying UMAP and clustering with {n_clusters} clusters...")
    combined_features_df = visualize_umap(dtw_distances, track_ids, combined_features_df, n_clusters)
    
    # Step 4: Count clusters by file
    cluster_counts = count_clusters_by_file(combined_features_df, file_to_track_ids, n_clusters)
    
    # Step 5: Create heatmap of cluster frequencies
    plot_cluster_heatmap(cluster_counts)
    
    # Optional: Create other visualizations
    create_feature_pairplot(combined_features_df)
    plot_radar_chart(combined_features_df, n_clusters)
    
    return all_tracks, combined_features_df, cluster_counts

def count_clusters_by_file(features_df, file_to_track_ids, n_clusters):
    """
    Count the number of tracks in each cluster for each file.
    
    Parameters:
    features_df (DataFrame): DataFrame with cluster assignments
    file_to_track_ids (dict): Mapping from file names to track IDs
    n_clusters (int): Number of clusters
    
    Returns:
    DataFrame: Cluster counts by file
    """
    # Initialize a DataFrame to store the counts
    files = list(file_to_track_ids.keys())
    clusters = list(range(n_clusters))
    
    counts = np.zeros((len(files), n_clusters))
    
    # Count tracks in each cluster for each file
    for i, file_name in enumerate(files):
        track_ids = file_to_track_ids[file_name]
        file_features = features_df.loc[track_ids]
        
        for j in clusters:
            counts[i, j] = sum(file_features['Cluster'] == j)
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(counts, index=files, columns=[f'Cluster {j}' for j in clusters])
    
    # Calculate percentages
    percentages_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100
    
    return {
        'counts': counts_df,
        'percentages': percentages_df
    }

def plot_cluster_heatmap(cluster_counts):
    """
    Create heatmaps showing both raw counts and percentages of tracks in each cluster by file.
    
    Parameters:
    cluster_counts (dict): Dictionary with 'counts' and 'percentages' DataFrames
    """
    # Plot raw counts
    plt.figure(figsize=(12, len(cluster_counts['counts']) * 0.8 + 2))
    ax = sns.heatmap(
        cluster_counts['counts'],
        cmap='coolwarm',
        annot=True,
        linewidths=0.5
    )
    plt.title('Number of Tracks in Each Cluster by File')
    plt.tight_layout()
    plt.savefig('cluster_counts_heatmap.svg', dpi=300)
    plt.show()
    
    # Plot percentages
    plt.figure(figsize=(12, len(cluster_counts['percentages']) * 0.8 + 2))
    ax = sns.heatmap(
        cluster_counts['percentages'],
        cmap='coolwarm',
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        vmin=0,
        vmax=100
    )
    plt.title('Percentage of Tracks in Each Cluster by File')
    plt.tight_layout()
    plt.savefig('cluster_percentages_heatmap.svg', dpi=300)
    plt.show()
    
    # Also create a stacked bar chart of percentages
    cluster_counts['percentages'].plot(
        kind='bar',
        stacked=True,
        figsize=(12, 8),
        colormap='viridis'
    )
    plt.title('Distribution of Clusters Across Files')
    plt.xlabel('File')
    plt.ylabel('Percentage')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cluster_distribution_bars.svg', dpi=300)
    plt.show()

# Enhanced main function to handle multiple files
def main_with_position_plots(file_paths=None, n_clusters=4):
    """
    Main function to run the analysis on multiple files.
    
    Parameters:
    file_paths (list): List of paths to Excel files with tracking data
    n_clusters (int): Number of clusters for k-means
    
    Returns:
    tuple: Combined tracks, features DataFrame with cluster assignments, and cluster counts by file
    """
    if file_paths is None:
        # Default to all Excel files in the current directory
        file_paths = [f for f in os.listdir() if f.endswith('.xlsx') and not f.startswith('~$')]
        file_paths = [os.path.abspath(f) for f in file_paths]
    
    all_tracks, features_df, cluster_counts = process_multiple_files(file_paths, n_clusters)
    
    print("\nPlotting tracks from position data...")
    plot_tracks_from_origin(file_paths)
    plot_tracks_by_cluster(file_paths, features_df, n_clusters)
    
    # Summarize clusters
    print("\nCluster Summary:")
    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['Cluster'] == cluster]
        print(f"\nCluster {cluster} ({len(cluster_data)} tracks):")
        print(cluster_data.drop('Cluster', axis=1).mean())
    
    return all_tracks, features_df, cluster_counts

# If running directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tracking data from multiple files using DTW and k-means clustering, and plot tracks')
    parser.add_argument('--files', nargs='+', help='Paths to Excel files with tracking data')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters for k-means')
    parser.add_argument('--directory', type=str, help='Directory containing Excel files to analyze')
    parser.add_argument('--plot-only', action='store_true', help='Only plot tracks without performing clustering')
    parser.add_argument('--max-tracks', type=int, default=100, help='Maximum number of tracks to plot per file')
    
    args = parser.parse_args()
    
    # Collect file paths
    if args.directory:
        # Get all Excel files in the specified directory
        file_paths = [os.path.join(args.directory, f) for f in os.listdir(args.directory) 
                     if f.endswith('.xlsx') and not f.startswith('~$')]
    elif args.files:
        file_paths = args.files
    else:
        # Default to Excel files in current directory
        file_paths = [f for f in os.listdir() if f.endswith('.xlsx') and not f.startswith('~$')]
    
    print(f"Found {len(file_paths)} files to analyze:")
    for f in file_paths:
        print(f"  - {os.path.basename(f)}")
    
    file_paths = ['Tracking files\\Treated.xlsx','Tracking files\\Untreated.xlsx']
    #file_paths = ['B:\\IRB-Groups\\SG\\Common\\CURRENT LAB MEMBERS\\Himanshu\\Shared_Kamil\\Imaris Data_Tracks plots_Action_Ades\\Tracks_Mindy support\\Compare_tracks\\1.xlsx']
    #file_paths = ['B:\\IRB-Groups\\SG\\Common\\CURRENT LAB MEMBERS\\Himanshu\\Shared_Kamil\\Imaris Data_Tracks plots_Action_Ades\\Tracks_Mindy support\\Condition2_Hematogeneous model\\1547.xlsx','B:\\IRB-Groups\\SG\\Common\\CURRENT LAB MEMBERS\\Himanshu\\Shared_Kamil\\Imaris Data_Tracks plots_Action_Ades\\Tracks_Mindy support\\Condition1_Lymphatic model\\Imaris format\\17 55.xlsx']    
    #file_paths = ['B:\\IRB-Groups\\SG\\Common\\CURRENT LAB MEMBERS\\Himanshu\\Shared_Kamil\\Imaris Data_Tracks plots_Action_Ades\\Tracks_Mindy support\\Condition3_NC\\Imaris format\\NC_classified_actions.xlsx']
    #file_paths = ['B:\\IRB-Groups\\SG\\Common\\CURRENT LAB MEMBERS\\Himanshu\\Shared_Kamil\\Imaris Data_Tracks plots_Action_Ades\\Tracks_Mindy support\\Condition1_Lymphatic model\\Imaris format\\1807.xlsx']
    all_tracks, features_df, cluster_counts = main_with_position_plots(file_paths, n_clusters=5)