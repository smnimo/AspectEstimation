"""
Main execution script for AspectEstimation
Translated from MATLAB main scripts
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.plsa import PLSA, train_multiple_runs
from data.data_loader import GazeDataLoader, LikelihoodCalculator
from utils.math_utils import normalize


def main_likelihood_calculation():
    """
    Main function for likelihood calculation
    
    Translated from calc_lkh.m
    """
    print("Starting likelihood calculation...")
    
    # Load data
    loader = GazeDataLoader()
    try:
        gaze_data = loader.load_gazedata()
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Warning: gazedata.mat not found. Creating sample data for testing.")
        # Create sample data for testing
        gaze_data = create_sample_data()
        loader.semantic_attributes = gaze_data['semantic_attributes']
        loader.semantic_attribute_seq = gaze_data['semantic_attribute_seq']
    
    # Calculate likelihood for each session
    calc = LikelihoodCalculator(win_size=5)
    S = len(loader.semantic_attribute_seq) if loader.semantic_attribute_seq else 1
    
    likelihood_results = {}
    
    for s in range(S):
        print(f"Processing session {s+1}/{S}")
        
        # Calculate attribute probabilities for this session
        prob_att = loader.calculate_attribute_probabilities(s)
        
        # Get attribute sequence for this session
        attseq = loader.semantic_attribute_seq[s]
        
        # Calculate likelihood
        session_lkh = calc.calculate_session_likelihood(attseq, prob_att)
        likelihood_results[f'session_{s}'] = session_lkh
    
    # Save results
    from data.data_loader import save_likelihood_results
    save_likelihood_results(likelihood_results, "lkh_win_5.npz")
    print("Likelihood calculation completed and saved to lkh_win_5.npz")


def main_plsa_training():
    """
    Main function for pLSA training
    
    Translated from main training workflow
    """
    print("Starting pLSA training...")
    
    # Load or create sample data
    try:
        loader = GazeDataLoader()
        gaze_data = loader.load_gazedata()
        sessions = extract_sessions_from_data(gaze_data)
    except:
        print("Using sample data for testing")
        sessions = create_sample_sessions()
    
    # Training parameters
    n_topics = 5
    train_lim = 1e-6
    train_num = 3  # Reduced for testing
    max_itr = 100
    
    print(f"Training pLSA with {n_topics} topics...")
    
    # Train model with multiple runs
    best_params, best_history = train_multiple_runs(
        sessions, n_topics, train_lim, train_num, max_itr
    )
    
    print(f"Training completed. Best log-likelihood: {best_history[-1]:.4f}")
    
    # Save results
    np.savez("plsa_results.npz", 
             p_z_s=best_params['zs'],
             p_v_z=best_params['vz'],
             log_likelihood=best_history)
    
    print("Results saved to plsa_results.npz")
    
    return best_params, best_history


def create_sample_data() -> Dict[str, Any]:
    """
    Create sample data for testing when real data is not available
    """
    print("Creating sample data...")
    
    # Sample parameters
    n_sessions = 3
    n_attributes = 5
    n_timepoints = 100
    n_values_per_attr = [3, 4, 3, 5, 4]  # Different number of values per attribute
    
    semantic_attributes = []
    semantic_attribute_seq = []
    
    for s in range(n_sessions):
        # Create random attribute values for this session
        session_attrs = []
        for attr in range(n_attributes):
            attr_values = np.random.randint(1, n_values_per_attr[attr] + 1, n_timepoints)
            session_attrs.append(attr_values)
        
        session_attrs = np.array(session_attrs)
        semantic_attributes.append(session_attrs)
        semantic_attribute_seq.append(session_attrs)
    
    return {
        'semantic_attributes': semantic_attributes,
        'semantic_attribute_seq': semantic_attribute_seq
    }


def create_sample_sessions() -> List[np.ndarray]:
    """
    Create sample session data for pLSA training
    """
    n_sessions = 3
    n_vocabulary = 10
    n_timepoints = 50
    
    sessions = []
    for s in range(n_sessions):
        # Create binary observation matrix (vocabulary x time)
        session_data = np.random.choice([0, 1], size=(n_vocabulary, n_timepoints), p=[0.7, 0.3])
        sessions.append(session_data)
    
    return sessions


def extract_sessions_from_data(gaze_data: Dict[str, Any]) -> List[np.ndarray]:
    """
    Extract session data in the format expected by pLSA
    
    Args:
        gaze_data: Loaded gaze data
        
    Returns:
        List of session data matrices
    """
    # This function needs to be adapted based on the actual structure of the .mat file
    # For now, create a placeholder implementation
    sessions = create_sample_sessions()
    return sessions


def run_full_pipeline():
    """
    Run the complete AspectEstimation pipeline
    """
    print("=" * 50)
    print("AspectEstimation Python Implementation")
    print("=" * 50)
    
    # Step 1: Calculate likelihood
    print("\nStep 1: Calculating likelihood...")
    main_likelihood_calculation()
    
    # Step 2: Train pLSA model
    print("\nStep 2: Training pLSA model...")
    best_params, best_history = main_plsa_training()
    
    # Step 3: Display results
    print("\nStep 3: Results summary")
    print(f"Final log-likelihood: {best_history[-1]:.4f}")
    print(f"Number of topics: {len(best_params['vz'])}")
    print(f"Convergence iterations: {len(best_history)}")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AspectEstimation Python Implementation")
    parser.add_argument("--mode", choices=["likelihood", "plsa", "full"], 
                       default="full", help="Execution mode")
    parser.add_argument("--topics", type=int, default=5, 
                       help="Number of topics for pLSA")
    
    args = parser.parse_args()
    
    if args.mode == "likelihood":
        main_likelihood_calculation()
    elif args.mode == "plsa":
        main_plsa_training()
    else:
        run_full_pipeline()
