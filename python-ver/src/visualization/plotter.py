"""
AspectEstimation Visualization Tools
Translates MATLAB plot_*.m series functionality to Python with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os
from PIL import Image


class AspectPlotter:
    """
    Main plotting class that replicates MATLAB plot_*.m functionality
    
    Handles:
    1. Likelihood plotting (plot_L_topic.m)
    2. Confusion matrix visualization (plot_conf_matrix.m)
    3. Duration analysis (plot_dur_usr.m)
    4. Evaluation metrics (plot_eval_AoF.m)
    5. Similarity analysis (plot_sim.m, plot_sim_task_asp.m)
    """
    
    def __init__(self, results_dir: str = "results", figure_dir: str = "figures"):
        """
        Initialize the plotter
        
        Args:
            results_dir: Directory containing result files (default: "results")
            figure_dir: Directory to save figures (default: "figures")
        """
        self.results_dir = results_dir
        self.figure_dir = figure_dir
        
        # Create figure directory if it doesn't exist
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        
        # Set matplotlib style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_likelihood_comparison(self, max_topics: int = 5, 
                                 save_path: str = None) -> plt.Figure:
        """
        Plot likelihood comparison across topics
        Translates: plot_L_topic.m
        
        Args:
            max_topics: Maximum number of topics to plot
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        # Load likelihood data from results
        L_all = np.zeros(max_topics)
        L_spe = np.zeros(max_topics)
        
        for topic in range(1, max_topics + 1):
            # Try to load results files
            all_file = os.path.join(self.results_dir, f"p_all_topic{topic}.npz")
            spe_file = os.path.join(self.results_dir, f"p_spe_topic{topic}.npz")
            
            try:
                if os.path.exists(all_file):
                    data = np.load(all_file, allow_pickle=True)
                    if 'L' in data:
                        L_all[topic-1] = data['L'][-1] if hasattr(data['L'], '__len__') else data['L']
                
                if os.path.exists(spe_file):
                    data = np.load(spe_file, allow_pickle=True)
                    if 'L' in data:
                        L_spe[topic-1] = data['L'][-1] if hasattr(data['L'], '__len__') else data['L']
            except Exception as e:
                print(f"Warning: Could not load topic {topic} data: {e}")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot likelihood curves
        topics = np.arange(1, max_topics + 1)
        ax1.plot(topics, L_all, 'o-', label='All data', linewidth=2, markersize=8)
        ax1.set_title('Likelihood: All Data')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Log-Likelihood')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(topics, L_spe, 's-', label='MSET', linewidth=2, markersize=8, color='orange')
        ax2.set_title('Likelihood: MSET')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Log-Likelihood')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved likelihood comparison to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, results_data: Dict[str, Any], 
                            topic: int = 3, save_path: str = None) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
        """
        Plot confusion matrices for task estimation
        Translates: plot_conf_matrix.m
        
        Args:
            results_data: Dictionary containing session data and results
            topic: Topic number to analyze
            save_path: Path to save figure (optional)
            
        Returns:
            Tuple of (Figure, MSET_confusion_matrix, no_MSET_confusion_matrix)
        """
        if 'session' not in results_data:
            raise ValueError("results_data must contain 'session' key")
        
        sessions = results_data['session']
        
        # Extract session structure
        n_task = 3  # Based on ReadwithTimestamp.m
        n_usr = len(set(s['usr'] for s in sessions.values()))
        
        # Create confusion matrices
        conf_mset = np.zeros((n_task, n_task + 1))
        conf_no_mset = np.zeros((n_task, n_task + 1))
        
        # Mock data for demonstration (in real implementation, use actual p.zs)
        # This would be replaced with actual trained model results
        np.random.seed(42)
        
        for task in range(1, n_task + 1):
            # Simulate task estimation results
            est_task_mset = np.random.choice(range(1, n_task + 2), n_usr)
            est_task_no_mset = np.random.choice(range(1, n_task + 2), n_usr)
            
            # Build confusion matrices
            for est in est_task_mset:
                if est <= n_task:
                    conf_mset[task-1, est-1] += 1
                else:
                    conf_mset[task-1, n_task] += 1
            
            for est in est_task_no_mset:
                if est <= n_task:
                    conf_no_mset[task-1, est-1] += 1
                else:
                    conf_no_mset[task-1, n_task] += 1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot MSET confusion matrix
        im1 = ax1.imshow(conf_mset, cmap='hot', aspect='auto')
        ax1.set_title('Task Estimation MSET: Confusion Matrix')
        ax1.set_xlabel('Estimated Task')
        ax1.set_ylabel('True Task')
        ax1.set_xticks(range(n_task + 1))
        ax1.set_xticklabels([f'Task {i+1}' for i in range(n_task)] + ['Unknown'])
        ax1.set_yticks(range(n_task))
        ax1.set_yticklabels([f'Task {i+1}' for i in range(n_task)])
        plt.colorbar(im1, ax=ax1)
        
        # Plot no-MSET confusion matrix
        im2 = ax2.imshow(conf_no_mset, cmap='hot', aspect='auto')
        ax2.set_title('Task Estimation w/o MSET: Confusion Matrix')
        ax2.set_xlabel('Estimated Task')
        ax2.set_ylabel('True Task')
        ax2.set_xticks(range(n_task + 1))
        ax2.set_xticklabels([f'Task {i+1}' for i in range(n_task)] + ['Unknown'])
        ax2.set_yticks(range(n_task))
        ax2.set_yticklabels([f'Task {i+1}' for i in range(n_task)])
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrices to {save_path}")
        
        return fig, conf_mset, conf_no_mset
    
    def plot_duration_by_user(self, sessions_data: Dict[str, Any], 
                            save_path: str = None) -> plt.Figure:
        """
        Plot duration analysis by user
        Translates: plot_dur_usr.m
        
        Args:
            sessions_data: Dictionary containing session data
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        if 'session' not in sessions_data:
            raise ValueError("sessions_data must contain 'session' key")
        
        sessions = sessions_data['session']
        
        # Extract duration data
        max_usr = max(s['usr'] for s in sessions.values())
        max_task = max(s['task'] for s in sessions.values())
        
        dur = np.zeros((max_usr, max_task))
        
        for (usr, task), session in sessions.items():
            if 'totalduration' in session:
                dur[usr-1, task-1] = session['totalduration']
        
        # Handle case where we only have one task
        if max_task == 1:
            # Add dummy data for visualization purposes
            dur = np.column_stack([dur, np.zeros((max_usr, 2))])
            max_task = 3
        
        # Users with incomplete data (from original MATLAB)
        usr_nocomp2 = [i for i in [28, 36] if i <= max_usr]  # Adjust for available users
        usr_nocomp1 = [i for i in [1, 3, 6, 18, 23, 35, 37] if i <= max_usr]  # Adjust for available users
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Sort by mean duration
        mean_dur = np.mean(dur, axis=1)
        sorted_idx_mean = np.argsort(mean_dur)
        
        # Plot boxplot sorted by mean
        bp1 = ax1.boxplot([dur[i, :] for i in sorted_idx_mean], 
                         labels=[str(i+1) for i in sorted_idx_mean],
                         patch_artist=True)
        
        # Color coding for incomplete users
        for i, idx in enumerate(sorted_idx_mean):
            if (idx + 1) in usr_nocomp2:
                bp1['boxes'][i].set_facecolor('red')
                bp1['boxes'][i].set_alpha(0.3)
            elif (idx + 1) in usr_nocomp1:
                bp1['boxes'][i].set_facecolor('yellow')
                bp1['boxes'][i].set_alpha(0.3)
        
        ax1.set_ylim([0, max(160, np.max(dur) * 1.1)])
        ax1.set_ylabel('Duration (s)')
        ax1.set_xlabel('Participant ID')
        ax1.set_title('Duration by User (sorted by mean)')
        ax1.grid(True, alpha=0.3)
        
        # Sort by median duration
        median_dur = np.median(dur, axis=1)
        sorted_idx_median = np.argsort(median_dur)
        
        # Plot boxplot sorted by median
        bp2 = ax2.boxplot([dur[i, :] for i in sorted_idx_median], 
                         labels=[str(i+1) for i in sorted_idx_median],
                         patch_artist=True)
        
        # Color coding for incomplete users
        for i, idx in enumerate(sorted_idx_median):
            if (idx + 1) in usr_nocomp2:
                bp2['boxes'][i].set_facecolor('red')
                bp2['boxes'][i].set_alpha(0.3)
            elif (idx + 1) in usr_nocomp1:
                bp2['boxes'][i].set_facecolor('yellow')
                bp2['boxes'][i].set_alpha(0.3)
        
        ax2.set_ylim([0, max(160, np.max(dur) * 1.1)])
        ax2.set_ylabel('Duration (s)')
        ax2.set_xlabel('Participant ID')
        ax2.set_title('Duration by User (sorted by median)')
        ax2.grid(True, alpha=0.3)
        
        # Add legend only if we have special users
        if usr_nocomp2 or usr_nocomp1:
            red_patch = patches.Patch(color='red', alpha=0.3, label='No completion 2')
            yellow_patch = patches.Patch(color='yellow', alpha=0.3, label='No completion 1')
            ax1.legend(handles=[red_patch, yellow_patch], loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved duration analysis to {save_path}")
        
        return fig
    
    def plot_aspect_matrices(self, results_data: Dict[str, Any], 
                           max_topics: int = 5, save_path: str = None) -> plt.Figure:
        """
        Plot aspect matrices for different topics
        Translates: aspect visualization from plot_L_topic.m
        
        Args:
            results_data: Dictionary containing results data
            max_topics: Maximum number of topics to plot
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, max_topics + 1, figsize=(18, 8))
        
        # Plot aspect matrices for each topic
        for topic in range(1, max_topics + 1):
            # Mock aspect data (replace with actual data loading)
            np.random.seed(42 + topic)
            asp_all = np.random.rand(12, 5)  # 12 items × 5 attributes
            asp_spe = np.random.rand(12, 5)
            
            # Plot all data aspects
            im1 = axes[0, topic-1].imshow(asp_all, cmap='gray', aspect='auto')
            axes[0, topic-1].set_title(f'All Data - Topic {topic}')
            axes[0, topic-1].set_ylabel('Items')
            axes[0, topic-1].set_xlabel('Attributes')
            
            # Plot MSET aspects
            im2 = axes[1, topic-1].imshow(asp_spe, cmap='gray', aspect='auto')
            axes[1, topic-1].set_title(f'MSET - Topic {topic}')
            axes[1, topic-1].set_ylabel('Items')
            axes[1, topic-1].set_xlabel('Attributes')
        
        # Plot ground truth (giventask)
        np.random.seed(42)
        giventask = np.random.rand(5, 12)  # Mock ground truth
        
        im3 = axes[0, max_topics].imshow(giventask.T, cmap='gray', aspect='auto')
        axes[0, max_topics].set_title('Ground Truth')
        axes[0, max_topics].set_ylabel('Items')
        axes[0, max_topics].set_xlabel('Attributes')
        
        im4 = axes[1, max_topics].imshow(giventask.T, cmap='gray', aspect='auto')
        axes[1, max_topics].set_title('Ground Truth')
        axes[1, max_topics].set_ylabel('Items')
        axes[1, max_topics].set_xlabel('Attributes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved aspect matrices to {save_path}")
        
        return fig
    
    def plot_similarity_analysis(self, sessions_data: Dict[str, Any], 
                               save_path: str = None) -> plt.Figure:
        """
        Plot similarity analysis between tasks and aspects
        Translates: plot_sim.m, plot_sim_task_asp.m
        
        Args:
            sessions_data: Dictionary containing session data
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        # Mock similarity data for demonstration
        np.random.seed(42)
        n_tasks = 3
        n_aspects = 5
        
        # Create similarity matrices
        task_similarity = np.random.rand(n_tasks, n_tasks)
        aspect_similarity = np.random.rand(n_aspects, n_aspects)
        task_aspect_similarity = np.random.rand(n_tasks, n_aspects)
        
        # Make symmetric
        task_similarity = (task_similarity + task_similarity.T) / 2
        aspect_similarity = (aspect_similarity + aspect_similarity.T) / 2
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Task similarity
        im1 = ax1.imshow(task_similarity, cmap='viridis', aspect='auto')
        ax1.set_title('Task Similarity Matrix')
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Task')
        ax1.set_xticks(range(n_tasks))
        ax1.set_xticklabels([f'Task {i+1}' for i in range(n_tasks)])
        ax1.set_yticks(range(n_tasks))
        ax1.set_yticklabels([f'Task {i+1}' for i in range(n_tasks)])
        plt.colorbar(im1, ax=ax1)
        
        # Aspect similarity
        im2 = ax2.imshow(aspect_similarity, cmap='viridis', aspect='auto')
        ax2.set_title('Aspect Similarity Matrix')
        ax2.set_xlabel('Aspect')
        ax2.set_ylabel('Aspect')
        ax2.set_xticks(range(n_aspects))
        ax2.set_xticklabels([f'Asp {i+1}' for i in range(n_aspects)])
        ax2.set_yticks(range(n_aspects))
        ax2.set_yticklabels([f'Asp {i+1}' for i in range(n_aspects)])
        plt.colorbar(im2, ax=ax2)
        
        # Task-Aspect relationship
        im3 = ax3.imshow(task_aspect_similarity, cmap='RdYlBu', aspect='auto')
        ax3.set_title('Task-Aspect Relationship')
        ax3.set_xlabel('Aspect')
        ax3.set_ylabel('Task')
        ax3.set_xticks(range(n_aspects))
        ax3.set_xticklabels([f'Asp {i+1}' for i in range(n_aspects)])
        ax3.set_yticks(range(n_tasks))
        ax3.set_yticklabels([f'Task {i+1}' for i in range(n_tasks)])
        plt.colorbar(im3, ax=ax3)
        
        # Correlation heatmap
        correlation_data = np.random.rand(n_tasks + n_aspects, n_tasks + n_aspects)
        correlation_data = (correlation_data + correlation_data.T) / 2
        
        im4 = ax4.imshow(correlation_data, cmap='coolwarm', aspect='auto')
        ax4.set_title('Overall Correlation Matrix')
        labels = [f'Task {i+1}' for i in range(n_tasks)] + [f'Asp {i+1}' for i in range(n_aspects)]
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.set_yticks(range(len(labels)))
        ax4.set_yticklabels(labels)
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved similarity analysis to {save_path}")
        
        return fig
    
    def plot_evaluation_metrics(self, evaluation_results: Dict[str, Any], 
                              save_path: str = None) -> plt.Figure:
        """
        Plot evaluation metrics
        Translates: plot_eval_AoF.m
        
        Args:
            evaluation_results: Dictionary containing evaluation data
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        # Mock evaluation data
        np.random.seed(42)
        n_topics = 5
        
        accuracy = np.random.rand(n_topics) * 0.3 + 0.6  # 0.6-0.9 range
        precision = np.random.rand(n_topics) * 0.3 + 0.5
        recall = np.random.rand(n_topics) * 0.3 + 0.5
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        topics = np.arange(1, n_topics + 1)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy plot
        ax1.plot(topics, accuracy, 'o-', linewidth=2, markersize=8, label='Accuracy')
        ax1.set_title('Classification Accuracy')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Precision plot
        ax2.plot(topics, precision, 's-', linewidth=2, markersize=8, 
                color='orange', label='Precision')
        ax2.set_title('Precision')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Precision')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Recall plot
        ax3.plot(topics, recall, '^-', linewidth=2, markersize=8, 
                color='green', label='Recall')
        ax3.set_title('Recall')
        ax3.set_xlabel('Number of Topics')
        ax3.set_ylabel('Recall')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # F1 Score plot
        ax4.plot(topics, f1_score, 'd-', linewidth=2, markersize=8, 
                color='red', label='F1 Score')
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Number of Topics')
        ax4.set_ylabel('F1 Score')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved evaluation metrics to {save_path}")
        
        return fig

    def plot_original_gaze_trajectories(self, 
                                       gaze_data: Dict[str, Any],
                                       image_path: Optional[str] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot original gaze trajectories (plot_org_gaze.m equivalent)
        
        Args:
            gaze_data: Dictionary containing gaze coordinates and metadata
                      Expected keys: 'gaze_seq', 'image_size', 'session_id'
            image_path: Optional background image path
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        try:
            # Load background image if provided
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    ax.imshow(img, alpha=0.7)
                    img_size = img.size  # (width, height)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
                    img_size = gaze_data.get('image_size', (1920, 1080))
            else:
                img_size = gaze_data.get('image_size', (1920, 1080))
                ax.set_xlim(0, img_size[0])
                ax.set_ylim(img_size[1], 0)  # Flip Y axis for screen coordinates
            
            # Extract gaze sequence
            if 'gaze_seq' in gaze_data:
                gaze_seq = np.array(gaze_data['gaze_seq'])
                
                # Convert normalized coordinates to pixel coordinates if needed
                if gaze_seq.max() <= 1.0:
                    gaze_seq = gaze_seq * np.array([img_size[0], img_size[1]])
                
                # Plot gaze trajectory
                ax.plot(gaze_seq[:, 0], gaze_seq[:, 1], 
                       'r-', linewidth=2, alpha=0.8, label='Gaze Trajectory')
                
                # Plot fixation points
                ax.scatter(gaze_seq[:, 0], gaze_seq[:, 1], 
                          c='blue', s=30, alpha=0.6, label='Fixation Points')
                
                # Add start/end markers
                if len(gaze_seq) > 0:
                    ax.scatter(gaze_seq[0, 0], gaze_seq[0, 1], 
                              c='green', s=100, marker='o', label='Start')
                    ax.scatter(gaze_seq[-1, 0], gaze_seq[-1, 1], 
                              c='red', s=100, marker='s', label='End')
                
            else:
                print("Warning: No 'gaze_seq' found in gaze_data")
            
            # Formatting
            session_id = gaze_data.get('session_id', 'Unknown')
            ax.set_title(f'Original Gaze Trajectory - Session {session_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set origin at top-left (standard screen coordinates)
            ax.set_ylim(img_size[1], 0)  # Y axis from bottom to top (inverted)
            ax.set_xlim(0, img_size[0])  # X axis from left to right
            
        except Exception as e:
            print(f"Error in plot_original_gaze_trajectories: {e}")
            ax.text(0.5, 0.5, f'Error plotting gaze data:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved gaze trajectory plot to {save_path}")
        
        return fig

    def plot_aspect_correlation_circles(self, 
                                       aspect_matrix: np.ndarray,
                                       feature_names: Optional[List[str]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot aspect correlation matrix with circles (plot_aspect.R equivalent)
        
        Args:
            aspect_matrix: Matrix of aspect correlations or feature loadings
            feature_names: Optional list of feature names for labeling
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        try:
            # Ensure matrix is 2D
            if aspect_matrix.ndim == 1:
                aspect_matrix = aspect_matrix.reshape(1, -1)
            
            # Take the second half of columns if specified (like R script)
            n_cols = aspect_matrix.shape[1]
            if n_cols > 6:  # If we have enough columns
                start_col = n_cols // 2 + 1
                matrix_subset = aspect_matrix[:, start_col:]
            else:
                matrix_subset = aspect_matrix
            
            # Normalize matrix to probability range [0, 1] if needed
            if matrix_subset.min() < 0 or matrix_subset.max() > 1:
                matrix_subset = (matrix_subset - matrix_subset.min()) / (matrix_subset.max() - matrix_subset.min())
            
            # Generate feature names if not provided
            if feature_names is None:
                feature_names = [f'Feature_{i+1}' for i in range(matrix_subset.shape[0])]
            
            # Create custom colormap (white to blue for probabilities)
            # 0 = white, 1 = blue
            colors = ['white', 'blue']
            n_bins = 200
            cmap = LinearSegmentedColormap.from_list('custom_wb', colors, N=n_bins)
            
            # Set fixed probability range [0, 1]
            vmin, vmax = 0.0, 1.0
            
            # Create circle-based correlation plot
            for i in range(matrix_subset.shape[0]):
                for j in range(matrix_subset.shape[1]):
                    value = matrix_subset[i, j]
                    
                    # Circle size proportional to absolute value (max = 1.0 for probabilities)
                    circle_size = abs(value) * 0.4
                    
                    # Color based on value
                    color = cmap((value - vmin) / (vmax - vmin))
                    
                    # Draw circle
                    circle = patches.Circle((j, i), circle_size, 
                                          facecolor=color, edgecolor='black', 
                                          linewidth=0.5)
                    ax.add_patch(circle)
            
            # Set axis properties
            ax.set_xlim(-0.5, matrix_subset.shape[1] - 0.5)
            ax.set_ylim(-0.5, matrix_subset.shape[0] - 0.5)
            
            # Set ticks and labels
            ax.set_xticks(range(matrix_subset.shape[1]))
            ax.set_yticks(range(matrix_subset.shape[0]))
            
            # Column labels (aspect numbers)
            col_labels = [f'Aspect_{i+1}' for i in range(matrix_subset.shape[1])]
            ax.set_xticklabels(col_labels, rotation=45, ha='right')
            
            # Row labels (feature names)
            if len(feature_names) >= matrix_subset.shape[0]:
                ax.set_yticklabels(feature_names[:matrix_subset.shape[0]])
            else:
                ax.set_yticklabels([f'Item_{i+1}' for i in range(matrix_subset.shape[0])])
            
            # Add colorbar
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([vmin, vmax])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Probability', rotation=270, labelpad=20)
            
            # Formatting
            ax.set_title('Aspect Correlation Matrix (Circle Plot)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Invert y-axis to match corrplot convention
            ax.invert_yaxis()
            
        except Exception as e:
            print(f"Error in plot_aspect_correlation_circles: {e}")
            ax.text(0.5, 0.5, f'Error plotting aspect correlations:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved aspect correlation plot to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, data: Dict[str, Any], 
                                  output_dir: str = None) -> List[str]:
        """
        Create a comprehensive visualization report
        
        Args:
            data: Dictionary containing all analysis data
            output_dir: Directory to save all plots (optional)
            
        Returns:
            List of saved figure paths
        """
        if output_dir is None:
            output_dir = self.figure_dir
        
        saved_paths = []
        
        # Likelihood comparison
        try:
            fig1 = self.plot_likelihood_comparison()
            path1 = os.path.join(output_dir, "likelihood_comparison.png")
            fig1.savefig(path1, dpi=300, bbox_inches='tight')
            saved_paths.append(path1)
            plt.close(fig1)
            print("✅ Created likelihood comparison")
        except Exception as e:
            print(f"❌ Error creating likelihood comparison: {e}")
        
        # Confusion matrices
        try:
            fig2, _, _ = self.plot_confusion_matrix(data)
            path2 = os.path.join(output_dir, "confusion_matrices.png")
            fig2.savefig(path2, dpi=300, bbox_inches='tight')
            saved_paths.append(path2)
            plt.close(fig2)
            print("✅ Created confusion matrices")
        except Exception as e:
            print(f"❌ Error creating confusion matrices: {e}")
        
        # Duration analysis
        try:
            fig3 = self.plot_duration_by_user(data)
            path3 = os.path.join(output_dir, "duration_analysis.png")
            fig3.savefig(path3, dpi=300, bbox_inches='tight')
            saved_paths.append(path3)
            plt.close(fig3)
            print("✅ Created duration analysis")
        except Exception as e:
            print(f"❌ Error creating duration analysis: {e}")
        
        # Aspect matrices
        try:
            fig4 = self.plot_aspect_matrices(data)
            path4 = os.path.join(output_dir, "aspect_matrices.png")
            fig4.savefig(path4, dpi=300, bbox_inches='tight')
            saved_paths.append(path4)
            plt.close(fig4)
            print("✅ Created aspect matrices")
        except Exception as e:
            print(f"❌ Error creating aspect matrices: {e}")
        
        # Similarity analysis
        try:
            fig5 = self.plot_similarity_analysis(data)
            path5 = os.path.join(output_dir, "similarity_analysis.png")
            fig5.savefig(path5, dpi=300, bbox_inches='tight')
            saved_paths.append(path5)
            plt.close(fig5)
            print("✅ Created similarity analysis")
        except Exception as e:
            print(f"❌ Error creating similarity analysis: {e}")
        
        # Evaluation metrics
        try:
            fig6 = self.plot_evaluation_metrics(data)
            path6 = os.path.join(output_dir, "evaluation_metrics.png")
            fig6.savefig(path6, dpi=300, bbox_inches='tight')
            saved_paths.append(path6)
            plt.close(fig6)
            print("✅ Created evaluation metrics")
        except Exception as e:
            print(f"❌ Error creating evaluation metrics: {e}")
        
        # Original gaze trajectories (plot_org_gaze.m equivalent)
        try:
            if 'gaze_trajectories' in data:
                gaze_data = data['gaze_trajectories']
                fig7 = self.plot_original_gaze_trajectories(gaze_data)
                path7 = os.path.join(output_dir, "gaze_trajectories.png")
                fig7.savefig(path7, dpi=300, bbox_inches='tight')
                saved_paths.append(path7)
                plt.close(fig7)
                print("✅ Created gaze trajectory plot")
        except Exception as e:
            print(f"❌ Error creating gaze trajectory plot: {e}")
        
        # Aspect correlation circles (plot_aspect.R equivalent)
        try:
            if 'params' in data and len(data['params']) > 0:
                params = data['params'][0]
                if 'vz' in params:
                    # Use the first topic's feature distribution
                    aspect_matrix = np.array(params['vz'][0])
                    fig8 = self.plot_aspect_correlation_circles(aspect_matrix)
                    path8 = os.path.join(output_dir, "aspect_correlations.png")
                    fig8.savefig(path8, dpi=300, bbox_inches='tight')
                    saved_paths.append(path8)
                    plt.close(fig8)
                    print("✅ Created aspect correlation circles")
        except Exception as e:
            print(f"❌ Error creating aspect correlation circles: {e}")
        
        print(f"\n🎨 Comprehensive report completed!")
        print(f"   Created {len(saved_paths)} out of 8 plots")
        print(f"   Saved to: {output_dir}")
        
        return saved_paths


def demo_visualization():
    """
    Demonstration of visualization capabilities
    """
    print("AspectEstimation Visualization Demo")
    print("=" * 50)
    
    # Create plotter
    plotter = AspectPlotter()
    
    # Mock data for demonstration
    mock_data = {
        'session': {
            (1, 1): {'usr': 1, 'task': 1, 'totalduration': 45.2},
            (1, 2): {'usr': 1, 'task': 2, 'totalduration': 52.1},
            (1, 3): {'usr': 1, 'task': 3, 'totalduration': 38.7},
            (2, 1): {'usr': 2, 'task': 1, 'totalduration': 67.3},
            (2, 2): {'usr': 2, 'task': 2, 'totalduration': 71.8},
            (2, 3): {'usr': 2, 'task': 3, 'totalduration': 55.4},
        }
    }
    
    # Create comprehensive report
    saved_paths = plotter.create_comprehensive_report(mock_data)
    
    print(f"Generated {len(saved_paths)} visualization plots:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    demo_visualization()
