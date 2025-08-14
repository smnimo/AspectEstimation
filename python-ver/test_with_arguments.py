"""
Test GazeDataReader with explicit file arguments
Demonstrates how to use specific file paths for knowledge.csv, gaze CSV, and LIST files
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.gaze_reader import GazeDataReader
from models.plsa import FourLayerPLSA


def test_with_explicit_files(knowledge_csv: str = None, 
                           gaze_csv: str = None, 
                           layout_list: str = None):
    """
    Test GazeDataReader with explicit file paths
    
    Args:
        knowledge_csv: Path to knowledge.csv file
        gaze_csv: Path to gaze data CSV file (e.g., 170116024150092.csv)
        layout_list: Path to layout LIST file (e.g., 170116024150092.list)
    """
    print("Testing GazeDataReader with explicit file arguments")
    print("=" * 60)
    
    # Initialize reader with knowledge.csv path
    reader = GazeDataReader(base_path=".", knowledge_csv_path=knowledge_csv)
    
    # Test 1: Load knowledge base
    print("\n1. Loading knowledge base...")
    if knowledge_csv:
        print(f"   Using explicit path: {knowledge_csv}")
    
    kb = reader.load_knowledge_base()
    print(f"   Knowledge base shape: {kb.shape}")
    print(f"   Sample entries:")
    print(kb[:3, :])
    
    # Test 2: Load specific session files if provided
    if gaze_csv and layout_list:
        print(f"\n2. Loading specific session files...")
        print(f"   Gaze CSV: {gaze_csv}")
        print(f"   Layout LIST: {layout_list}")
        
        try:
            session = reader.load_session_from_files(
                csv_file=gaze_csv,
                list_file=layout_list,
                user_id=1,
                task_id=1
            )
            
            if session:
                print(f"   ✅ Session loaded successfully")
                print(f"   Item sequence length: {len(session['item_sequence'])}")
                print(f"   Semantic sequence shape: {session['semantic_attribute_seq'].shape}")
                print(f"   Sample item sequence: {session['item_sequence'][:10]}")
                
                # Test with four-layer pLSA
                print(f"\n3. Testing four-layer pLSA with real session...")
                
                # Create Feature→Observation mapping from knowledge base
                n_items, n_attributes = kb.shape
                n_values = np.max(kb, axis=0)
                total_features = np.sum(n_values)
                
                # Create P(o|v) mapping
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
                
                # Train model
                model = FourLayerPLSA(n_topics=2)
                model.set_observation_model(p_o_v)
                
                sessions = [session['semantic_attribute_seq']]
                params, history = model.fit(sessions, train_lim=1e-4, max_itr=20)
                
                print(f"   Training completed: {len(history)} iterations")
                print(f"   Final likelihood: {history[-1]:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error loading session: {e}")
    else:
        print(f"\n2. No specific session files provided - using sample data")
        
        # Create sample session with loaded knowledge base
        print(f"   Creating sample session with real knowledge base...")
        
        n_items, n_attributes = kb.shape
        n_timepoints = 100
        
        # Create random item sequence
        item_sequence = np.random.randint(1, n_items + 1, n_timepoints)
        
        # Convert to semantic attributes using real knowledge base
        semantic_seq = reader.items_to_semantic_attributes(item_sequence)
        
        print(f"   Sample session created:")
        print(f"   - Item sequence length: {len(item_sequence)}")
        print(f"   - Semantic sequence shape: {semantic_seq.shape}")
        print(f"   - Sample items: {item_sequence[:10]}")
    
    print(f"\n" + "=" * 60)
    print(f"Test completed!")
    return reader, kb


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Test GazeDataReader with explicit file paths')
    
    parser.add_argument('--knowledge', '-k', type=str, default=None,
                        help='Path to knowledge.csv file')
    parser.add_argument('--gaze', '-g', type=str, default=None,
                        help='Path to gaze data CSV file (e.g., 170116024150092.csv)')
    parser.add_argument('--layout', '-l', type=str, default=None,
                        help='Path to layout LIST file (e.g., 170116024150092.list)')
    
    args = parser.parse_args()
    
    # Run test with provided arguments
    reader, kb = test_with_explicit_files(
        knowledge_csv=args.knowledge,
        gaze_csv=args.gaze,
        layout_list=args.layout
    )
    
    return reader, kb


if __name__ == "__main__":
    print("Usage examples:")
    print("python test_with_arguments.py --knowledge ../knowledge.csv")
    print("python test_with_arguments.py --knowledge ../knowledge.csv --gaze ../170116024150092.csv --layout ../170116024150092.list")
    print("")
    
    reader, kb = main()
