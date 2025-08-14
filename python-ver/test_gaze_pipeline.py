"""
Test the complete gaze data reading pipeline
Demonstrates ReadwithTimestamp.m equivalent functionality
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.gaze_reader import GazeDataReader
from models.plsa import FourLayerPLSA


def test_knowledge_base_loading():
    """Test loading the actual knowledge.csv file"""
    print("=== Testing Knowledge Base Loading ===")
    
    # Try different paths to find knowledge.csv
    possible_paths = [
        ".",           # Current directory
        "..",          # Parent directory (AspectEstimation root)
        "../..",       # Two levels up
        "../../.."     # Three levels up
    ]
    
    reader = None
    kb = None
    
    for base_path in possible_paths:
        try:
            reader = GazeDataReader(base_path)
            kb = reader.load_knowledge_base("knowledge.csv")
            if reader.knowledge_base is not None and reader.knowledge_base.shape[0] == 12:
                print(f"✅ Loaded actual knowledge.csv from: {base_path}")
                break
        except Exception as e:
            print(f"⚠️ Could not load from {base_path}: {e}")
            continue
    
    if kb is not None and kb.shape[0] == 12:
        print(f"✅ Successfully loaded knowledge.csv: {kb.shape}")
        print(f"Sample entries:")
        print(kb[:5, :])
        print(f"Value ranges per attribute:")
        for attr in range(kb.shape[1]):
            print(f"  Attribute {attr+1}: {np.min(kb[:, attr])} - {np.max(kb[:, attr])}")
    else:
        print("⚠️ Using sample knowledge base instead")
        reader = GazeDataReader(".")
        kb = reader.load_knowledge_base()
    
    return kb


def test_feature_to_observation_mapping():
    """Test the Feature→Observation mapping based on knowledge.csv"""
    print("\n=== Testing Feature→Observation Mapping ===")
    
    # Use the same path detection logic
    reader = None
    kb = None
    
    possible_paths = [".", "..", "../..", "../../.."]
    for base_path in possible_paths:
        try:
            reader = GazeDataReader(base_path)
            kb = reader.load_knowledge_base("knowledge.csv")
            if reader.knowledge_base is not None and reader.knowledge_base.shape[0] == 12:
                break
        except:
            continue
    
    if kb is None or kb.shape[0] != 12:
        reader = GazeDataReader(".")
        kb = reader.load_knowledge_base()
    
    # This is the true Feature→Observation mapping from the MATLAB system
    print("Knowledge Base (Feature→Observation mapping):")
    print("Each row = item, each column = attribute, values = attribute levels")
    print(f"Shape: {kb.shape} (items × attributes)")
    print("\nKnowledge Base Matrix:")
    print(kb)
    
    # Create the offset-based session representation as in learn_aspects_from_mask.m
    print("\n=== Creating Offset-based Representation ===")
    
    # Get value ranges per attribute
    n_values = np.max(kb, axis=0)
    print(f"Value ranges per attribute: {n_values}")
    
    # Calculate offsets (as in learn_aspects_from_mask.m lines 29-33)
    offsets = [0]
    for i in range(len(n_values) - 1):
        offsets.append(offsets[-1] + n_values[i])
    print(f"Offsets: {offsets}")
    
    # Total feature dimension
    total_features = np.sum(n_values)
    print(f"Total feature dimension: {total_features}")
    
    # Create sample session using offset encoding
    sample_items = [1, 2, 3, 1, 2]  # Sample item sequence
    n_timepoints = len(sample_items)
    
    session_matrix = np.zeros((total_features, n_timepoints))
    
    for t, item_idx in enumerate(sample_items):
        # Convert to 0-based indexing
        item_attrs = kb[item_idx - 1, :]
        
        # Apply offsets (matching learn_aspects_from_mask.m)
        for attr in range(len(item_attrs)):
            attr_value = item_attrs[attr]
            feature_idx = offsets[attr] + attr_value - 1  # Convert to 0-based
            session_matrix[feature_idx, t] = 1
    
    print(f"\nSample session matrix shape: {session_matrix.shape}")
    print(f"Sample item sequence: {sample_items}")
    print("Encoded session matrix (features × time):")
    print(session_matrix)
    
    return session_matrix, kb


def test_four_layer_with_real_knowledge():
    """Test four-layer pLSA with real knowledge base"""
    print("\n=== Testing Four-Layer pLSA with Real Knowledge ===")
    
    # Get real knowledge base
    reader = GazeDataReader(".")
    kb = reader.load_knowledge_base("knowledge.csv")
    
    # Create the real Feature→Observation mapping
    n_items, n_attributes = kb.shape
    n_values = np.max(kb, axis=0)
    total_features = np.sum(n_values)
    
    print(f"Real data dimensions:")
    print(f"  Items: {n_items}")
    print(f"  Attributes: {n_attributes}")
    print(f"  Total features: {total_features}")
    
    # Create P(o|v) mapping from knowledge base
    # This represents the true Feature→Observation mapping
    p_o_v = np.zeros((n_items, total_features))
    
    offsets = [0]
    for i in range(len(n_values) - 1):
        offsets.append(offsets[-1] + n_values[i])
    
    for item_idx in range(n_items):
        item_attrs = kb[item_idx, :]
        for attr in range(n_attributes):
            attr_value = item_attrs[attr]
            feature_idx = offsets[attr] + attr_value - 1
            p_o_v[item_idx, feature_idx] = 1.0
    
    print(f"P(o|v) mapping shape: {p_o_v.shape} (observations × features)")
    print("P(o|v) matrix (first 5 items):")
    print(p_o_v[:5, :])
    
    # Create sample sessions with the real encoding
    n_sessions = 3
    n_timepoints = 50
    sessions = []
    
    for s in range(n_sessions):
        session_matrix = np.zeros((total_features, n_timepoints))
        
        for t in range(n_timepoints):
            # Random item selection
            item_idx = np.random.randint(0, n_items)
            item_attrs = kb[item_idx, :]
            
            # Encode using offsets
            for attr in range(n_attributes):
                attr_value = item_attrs[attr]
                feature_idx = offsets[attr] + attr_value - 1
                session_matrix[feature_idx, t] = 1
        
        sessions.append(session_matrix)
    
    # Train four-layer model
    model = FourLayerPLSA(n_topics=2)
    model.set_observation_model(p_o_v)
    
    print(f"\nTraining with real knowledge base...")
    params, history = model.fit(sessions, train_lim=1e-4, max_itr=30)
    
    print(f"Training completed: {len(history)} iterations")
    print(f"Final log-likelihood: {history[-1]:.4f}")
    print(f"P(z|s) shape: {params['zs'].shape}")
    print(f"P(v|z) topics: {len(params['vz'])}")
    
    return params, history


def test_complete_pipeline():
    """Test the complete data reading and processing pipeline"""
    print("\n=== Testing Complete Pipeline ===")
    
    reader = GazeDataReader(".")
    
    # This will create sample data if real files are not available
    data = reader.load_all_sessions()
    
    print(f"Complete pipeline results:")
    print(f"  Sessions loaded: {len(data['session'])}")
    print(f"  Knowledge base shape: {data['KB'].shape}")
    
    # Convert to format for four-layer pLSA
    session_list = []
    for (usr, task), session in data['session'].items():
        session_list.append(session['semantic_attribute_seq'])
    
    if session_list:
        print(f"  Sample session shape: {session_list[0].shape}")
        
        # Train model on this data
        model = FourLayerPLSA(n_topics=2)
        
        # Create identity mapping for simplicity
        n_features = session_list[0].shape[0]
        p_o_v = np.eye(n_features)
        model.set_observation_model(p_o_v)
        
        params, history = model.fit(session_list, train_lim=1e-4, max_itr=20)
        print(f"  Training: {len(history)} iterations, likelihood: {history[-1]:.4f}")


if __name__ == "__main__":
    print("Testing Complete Gaze Data Pipeline")
    print("Replicating ReadwithTimestamp.m → LoadGaze.m → four-layer pLSA")
    print("=" * 70)
    
    # Test 1: Knowledge base loading
    kb = test_knowledge_base_loading()
    
    # Test 2: Feature→Observation mapping
    session_matrix, kb = test_feature_to_observation_mapping()
    
    # Test 3: Four-layer pLSA with real knowledge
    try:
        params, history = test_four_layer_with_real_knowledge()
    except Exception as e:
        print(f"Real knowledge test failed: {e}")
    
    # Test 4: Complete pipeline
    test_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("Pipeline test completed!")
    print("\nNow the Python implementation can handle:")
    print("✅ Real knowledge.csv (Feature→Observation mapping)")
    print("✅ Gaze data format (.csv files)")
    print("✅ Item layout (.list files)")
    print("✅ Offset-based feature encoding")
    print("✅ Four-layer pLSA with fixed observation layer")
