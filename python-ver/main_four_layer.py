"""
Main execution script for Four-Layer AspectEstimation
Session → Topic → Feature → Observation
Feature → Observation mapping is externally p    print("Four-layer pLSA demonstration completed!")
    return results


def save_results(results: Dict, savedir: str = 'results_four_layer'):ed and fixed (not learned)
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.plsa import FourLayerPLSA


def create_sample_observation_mapping(n_features: int) -> np.ndarray:
    """
    Create a sample Feature → Observation mapping
    
    In the four-layer model, this mapping is externally provided and fixed.
    For demonstration, we create an identity mapping.
    
    Args:
        n_features: Number of features
        
    Returns:
        P(o|v) matrix (observations x features)
    """
    return np.eye(n_features)


def create_sample_sessions(n_sessions: int = 5, n_features: int = 15, 
                          n_timepoints: int = 30) -> List[np.ndarray]:
    """
    Create sample session data for testing the four-layer model
    
    Args:
        n_sessions: Number of sessions
        n_features: Number of features 
        n_timepoints: Number of time points per session
        
    Returns:
        List of session matrices (features x time)
    """
    sessions = []
    
    for s in range(n_sessions):
        # Create one-hot encoded feature data
        session_data = np.zeros((n_features, n_timepoints))
        for t in range(n_timepoints):
            # Each time point activates exactly one feature
            feature_idx = np.random.randint(0, n_features)
            session_data[feature_idx, t] = 1
        sessions.append(session_data)
    
    return sessions


def train_four_layer_model(sessions: List[np.ndarray], n_topics: int,
                           p_o_v: np.ndarray = None, max_itr: int = 100) -> Dict:
    """
    Train the four-layer pLSA model
    
    Args:
        sessions: List of session data (features x time)
        n_topics: Number of topics to learn
        p_o_v: Feature → Observation mapping (if None, uses identity)
        max_itr: Maximum iterations
        
    Returns:
        Trained model parameters
    """
    # Initialize model
    model = FourLayerPLSA(n_topics)
    
    # Set the fixed observation model
    if p_o_v is None:
        n_features = sessions[0].shape[0] if sessions else 10
        p_o_v = create_sample_observation_mapping(n_features)
    
    model.set_observation_model(p_o_v)
    
    # Train model
    print(f"Training four-layer pLSA with {n_topics} topics...")
    params, history = model.fit(sessions, train_lim=1e-4, max_itr=max_itr)
    
    print(f"Training completed in {len(history)} iterations")
    if history:
        print(f"Final log-likelihood: {history[-1]:.4f}")
    
    return params, history


def main_four_layer_demo():
    """
    Demonstrate the four-layer pLSA implementation
    Session → Topic → Feature → Observation
    """
    print("=" * 60)
    print("Four-Layer pLSA Demonstration")
    print("Session → Topic → Feature → Observation")
    print("Feature → Observation is externally fixed")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample data...")
    sessions = create_sample_sessions(n_sessions=5, n_features=15, n_timepoints=30)
    print(f"Created {len(sessions)} sessions")
    print(f"Each session shape: {sessions[0].shape} (features x time)")
    
    # Create observation mapping
    n_features = sessions[0].shape[0]
    p_o_v = create_sample_observation_mapping(n_features)
    print(f"Observation mapping shape: {p_o_v.shape} (observations x features)")
    print("Using identity mapping: P(o|v) = δ(o,v)")
    
    # Train for different numbers of topics
    results = {}
    for n_topics in [1, 2, 3]:
        print(f"\n--- Training with {n_topics} topics ---")
        params, history = train_four_layer_model(
            sessions, n_topics, p_o_v, max_itr=50
        )
        
        results[n_topics] = {
            'params': params,
            'history': history,
            'final_likelihood': history[-1] if history else None
        }
        
        # Display parameter information
        print(f"P(z|s) shape: {params['zs'].shape}")  # (sessions x topics)
        print(f"P(v|z) distributions: {len(params['vz'])}")  # list of topic→feature
        if params['vz']:
            print(f"Each P(v|z) shape: {params['vz'][0].shape}")  # (features,)
    
    # Compare results
    print(f"\n--- Results Summary ---")
    for n_topics, result in results.items():
        likelihood = result['final_likelihood']
        iterations = len(result['history'])
        print(f"Topics: {n_topics}, Final likelihood: {likelihood:.4f}, Iterations: {iterations}")
    
    print("\nFour-layer pLSA demonstration completed!")
    return results


def save_results(results: Dict, savedir: str = 'results_true_four_layer'):
    """
    Save the training results
    
    Args:
        results: Training results dictionary
        savedir: Directory to save results
    """
    os.makedirs(savedir, exist_ok=True)
    
    for n_topics, result in results.items():
        # Save parameters
        params_file = f'{savedir}/params_topic{n_topics}.npz'
        np.savez(params_file, **result['params'], history=result['history'])
        
        # Save P(v|z) as CSV for analysis
        if 'vz' in result['params']:
            vz_matrix = np.hstack([v.reshape(-1, 1) for v in result['params']['vz']])
            csv_file = f'{savedir}/P_v_z_topic{n_topics}.csv'
            np.savetxt(csv_file, vz_matrix, delimiter=',')
            
        print(f"Results for {n_topics} topics saved to {savedir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Four-Layer pLSA Demonstration")
    parser.add_argument("--sessions", type=int, default=5, help="Number of sessions")
    parser.add_argument("--features", type=int, default=15, help="Number of features")
    parser.add_argument("--max-topics", type=int, default=3, help="Maximum topics to train")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    
    args = parser.parse_args()
    
    # Run demonstration
    results = main_four_layer_demo()
    
    # Save results if requested
    if args.save:
        save_results(results)
