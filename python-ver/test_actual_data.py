"""
Test the gaze data reader with actual files using relative paths
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.gaze_reader import GazeDataReader


def test_actual_data_loading():
    """Test loading actual data files using relative paths"""
    print("Testing actual data loading with relative paths...")
    
    # Initialize reader pointing to sample_data directory
    reader = GazeDataReader(knowledge_csv_path='../sample_data/knowledge.csv')
    
    # Try to load knowledge base
    print("\n1. Loading knowledge base...")
    kb = reader.load_knowledge_base()
    print(f"Knowledge base shape: {kb.shape}")
    print(f"Knowledge base sample:\n{kb[:3, :]}")
    
    # Check for available data files
    print("\n2. Checking for available data files...")
    base_path = "../sample_data"
    
    csv_files = []
    list_files = []
    
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith('.csv') and file != 'knowledge.csv':
                csv_files.append(file)
            elif file.endswith('.list'):
                list_files.append(file)
    
    print(f"Found CSV files: {csv_files}")
    print(f"Found LIST files: {list_files}")
    
    # Try to process available data
    print("\n3. Processing available data...")
    data = reader.load_all_sessions()
    
    print(f"\nResults:")
    print(f"- Knowledge base: {data['KB'].shape}")
    print(f"- Sessions: {len(data['session'])}")
    print(f"- Session keys: {list(data['session'].keys())}")
    
    # Display first session details if available
    if data['session']:
        first_session_key = list(data['session'].keys())[0]
        first_session = data['session'][first_session_key]
        print(f"\nFirst session details (User {first_session_key[0]}, Task {first_session_key[1]}):")
        print(f"- Item sequence length: {len(first_session['f'])}")
        print(f"- Semantic sequence shape: {first_session['semantic_attribute_seq'].shape}")
        print(f"- Sample item sequence: {first_session['f'][:10]}")
    
    return data


def test_specific_files():
    """Test with specific known files"""
    print("\n" + "="*60)
    print("Testing with specific files...")
    
    reader = GazeDataReader(knowledge_csv_path='../sample_data/knowledge.csv')
    
    # Test files that should exist
    csv_file = "../sample_data/170116024150092.csv"
    list_file = "../sample_data/170116024150092.list"
    
    if os.path.exists(csv_file) and os.path.exists(list_file):
        print(f"Found test files:")
        print(f"- CSV: {csv_file}")
        print(f"- LIST: {list_file}")
        
        try:
            # Load knowledge base first
            kb = reader.load_knowledge_base()
            
            # Process the specific session using load_session_from_files
            session_data = reader.load_session_from_files(csv_file, list_file)
            
            print(f"\nProcessed session data:")
            print(f"- Item sequence length: {len(session_data['f'])}")
            print(f"- Semantic sequence shape: {session_data['semantic_attribute_seq'].shape}")
            print(f"- Total duration: {session_data['totalduration']:.2f} seconds")
            print(f"- Task: {session_data['task']}")
            print(f"- User: {session_data['usr']}")
            
            # Show sample data
            print(f"\nSample item sequence (first 20): {session_data['f'][:20]}")
            print(f"Sample semantic attributes (first 5 timepoints):")
            print(session_data['semantic_attribute_seq'][:, :5])
            
        except Exception as e:
            print(f"Error processing files: {e}")
    else:
        print("Test files not found in expected locations")


def test_actual_plsa_training():
    """Test pLSA training with actual data and save results"""
    print("\n" + "="*60)
    print("Testing pLSA Training with Real Data...")
    
    reader = GazeDataReader(knowledge_csv_path='../sample_data/knowledge.csv')
    
    # Test files
    csv_file = "../sample_data/170116024150092.csv"
    list_file = "../sample_data/170116024150092.list"
    
    if os.path.exists(csv_file) and os.path.exists(list_file):
        print(f"Training pLSA with real data...")
        
        try:
            # Load session data
            session_data = reader.load_session_from_files(csv_file, list_file)
            
            # Import pLSA model
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from src.models.plsa import FourLayerPLSA
            
            # Use 15D one-hot features
            onehot_features = session_data['semantic_attribute_seq_onehot']
            print(f"Training with 15D one-hot features: {onehot_features.shape}")
            
            # Create results directory
            os.makedirs("results", exist_ok=True)
            
            # Train models with different topic numbers
            for n_topics in [2, 3]:
                print(f"\nTraining {n_topics}-topic model...")
                
                model = FourLayerPLSA(n_topics=n_topics)
                
                # Set identity observation model for 15D features
                n_features = onehot_features.shape[0]  # 15
                p_o_v = np.eye(n_features)
                model.set_observation_model(p_o_v)
                
                # Train with single session
                sessions = [onehot_features]
                params, likelihood_history = model.fit(sessions, max_itr=50)
                
                # Save results
                result_file = f"results/test_real_data_topic{n_topics}.npz"
                
                np.savez(result_file,
                        L=np.array([likelihood_history[-1]]),
                        params=params,
                        history=likelihood_history,
                        allow_pickle=True)
                
                print(f"✅ Training completed: L={likelihood_history[-1]:.2f}, saved to {result_file}")
            
            return True
            
        except Exception as e:
            print(f"Error in pLSA training: {e}")
            return False
    else:
        print("Real data files not found, skipping pLSA training")
        return False


if __name__ == "__main__":
    print("Testing GazeDataReader with actual files")
    print("="*60)
    
    # Test 1: General data loading
    data = test_actual_data_loading()
    
    # Test 2: Specific files
    test_specific_files()
    
    # Test 3: pLSA training with real data
    test_actual_plsa_training()
    
    print("\nTesting completed!")
