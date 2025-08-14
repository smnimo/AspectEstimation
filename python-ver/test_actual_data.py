"""
Test the gaze data reader with actual files using relative paths
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.gaze_reader import GazeDataReader


def test_actual_data_loading():
    """Test loading actual data files using relative paths"""
    print("Testing actual data loading with relative paths...")
    
    # Initialize reader pointing to AspectEstimation root
    reader = GazeDataReader("../..")
    
    # Try to load knowledge base
    print("\n1. Loading knowledge base...")
    kb = reader.load_knowledge_base()
    print(f"Knowledge base shape: {kb.shape}")
    print(f"Knowledge base sample:\n{kb[:3, :]}")
    
    # Check for available data files
    print("\n2. Checking for available data files...")
    base_path = "../.."
    
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
    
    reader = GazeDataReader("../..")
    
    # Test files that should exist
    csv_file = "../../170116024150092.csv"
    list_file = "../../170116024150092.list"
    
    if os.path.exists(csv_file) and os.path.exists(list_file):
        print(f"Found test files:")
        print(f"- CSV: {csv_file}")
        print(f"- LIST: {list_file}")
        
        try:
            # Load knowledge base first
            kb = reader.load_knowledge_base()
            
            # Process the specific session
            session_data = reader.process_session(csv_file, list_file, 1, 1)
            
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


if __name__ == "__main__":
    print("Testing GazeDataReader with actual files")
    print("="*60)
    
    # Test 1: General data loading
    data = test_actual_data_loading()
    
    # Test 2: Specific files
    test_specific_files()
    
    print("\nTesting completed!")
