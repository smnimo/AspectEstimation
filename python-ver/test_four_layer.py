"""
Test the corrected four-layer pLSA implementation
Session → Topic → Feature → Observation with fixed observation layer
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.plsa import FourLayerPLSA
from src.models.segments import SegmentCalculator


def create_test_observation_model(n_features: int, n_observations: int) -> np.ndarray:
    """Create a test Feature → Observation mapping"""
    # Create simple 1-to-1 mapping for testing
    if n_features == n_observations:
        return np.eye(n_features)
    else:
        # Create random but fixed mapping
        p_o_v = np.random.rand(n_observations, n_features)
        # Normalize each feature to sum to 1 over observations
        for v in range(n_features):
            p_o_v[:, v] = p_o_v[:, v] / np.sum(p_o_v[:, v])
        return p_o_v


def test_simple_four_layer():
    """Test four-layer pLSA with simple synthetic data"""
    print("Testing four-layer pLSA implementation...")
    
    # Create simple test data
    n_sessions = 3
    n_observations = 10
    n_features = 10
    n_topics = 2
    
    # Create Feature → Observation mapping (externally fixed)
    p_o_v = create_test_observation_model(n_features, n_observations)
    print(f"Created observation model: {p_o_v.shape}")
    
    # Create simple sessions (already in observation space)
    sessions = []
    for s in range(n_sessions):
        # Create one-hot encoded observation data
        session_data = np.zeros((n_observations, 20))  # 20 time points
        for t in range(20):
            obs_idx = np.random.randint(0, n_observations)
            session_data[obs_idx, t] = 1
        sessions.append(session_data)
    
    print(f"Created {len(sessions)} test sessions")
    
    # Initialize and train model
    model = FourLayerPLSA(n_topics)
    model.set_observation_model(p_o_v)
    
    print("Training model...")
    params, history = model.fit(sessions, train_lim=1e-4, max_itr=20)
    
    print(f"Training completed in {len(history)} iterations")
    print(f"Final log-likelihood: {history[-1]:.4f}")
    
    # Check parameter shapes
    print(f"P(z|s) shape: {params['zs'].shape}")  # (sessions, topics)
    print(f"P(v|z) length: {len(params['vz'])}")  # list of topic distributions
    if params['vz']:
        print(f"P(v|z) feature shape: {params['vz'][0].shape}")  # (features,)
    
    return model, params, history


def test_with_segments():
    """Test with segment data using the segments.py module"""
    print("\nTesting with segment data...")
    
    # Create simple test sessions for the corrected architecture
    n_sessions = 5
    n_features = 15  # Feature dimension
    sessions = []
    
    for s in range(n_sessions):
        # Create one-hot encoded feature data (features x time)
        session_data = np.zeros((n_features, 30))  # 30 time points
        for t in range(30):
            feature_idx = np.random.randint(0, n_features)
            session_data[feature_idx, t] = 1
        sessions.append(session_data)
    
    print(f"Created {len(sessions)} test sessions")
    if sessions:
        print(f"First session shape: {sessions[0].shape}")
    
    # Train for different numbers of topics
    for n_topics in [1, 2]:
        print(f"\nTraining with {n_topics} topics...")
        
        model = FourLayerPLSA(n_topics)
        
        # Create identity observation model (Feature = Observation for simplicity)
        if sessions:
            n_features = sessions[0].shape[0]
            p_o_v = np.eye(n_features)  # Identity: P(o|v) = δ(o,v)
            model.set_observation_model(p_o_v)
            
            params, history = model.fit(sessions, train_lim=1e-4, max_itr=20)
            
            print(f"Training completed: {len(history)} iterations")
            if history:
                print(f"Final log-likelihood: {history[-1]:.4f}")
                
            # Save results to results directory
            os.makedirs("results", exist_ok=True)
            result_file = f"results/test_synthetic_topic{n_topics}.npz"
            
            np.savez(result_file,
                    L=np.array([history[-1]]),
                    params=params,
                    history=history,
                    allow_pickle=True)
            
            print(f"Results saved to: {result_file}")
                
            # Display parameter shapes
            print(f"P(z|s) shape: {params['zs'].shape}")
            print(f"P(v|z) topics: {len(params['vz'])}")


if __name__ == "__main__":
    print("Testing four-layer pLSA implementation")
    print("Architecture: Session → Topic → Feature → Observation")
    print("Feature → Observation is externally fixed\n")
    
    # Test 1: Simple synthetic data
    model1, params1, history1 = test_simple_four_layer()
    
    # Test 2: With segment data
    test_with_segments()
    
    print("\nAll tests completed successfully!")
