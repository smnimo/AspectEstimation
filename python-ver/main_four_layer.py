"""
Main execution script for Four-Layer AspectEstimation
Implements the custom 4-layer model: Session -> Segment -> Topic -> Vocabulary
Translated from learn_aspects_from_mask.m and related MATLAB scripts
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.plsa import FourLayerPLSA, train_multiple_runs
from models.segments import SegmentCalculator, SessionProcessor
from data.data_loader import GazeDataLoader, LikelihoodCalculator, save_likelihood_results, load_likelihood_results
from utils.math_utils import normalize


def main_learn_aspects_from_mask(train_runs: int = 2):
    """
    Main function implementing the four-layer aspect learning pipeline
    
    Args:
        train_runs: Number of training runs per topic
    
    Translated from learn_aspects_from_mask.m
    """
    print("Starting Four-Layer Aspect Learning...")
    start_time = time.time()
    
    # Parameters (matching MATLAB script)
    win_size = 5
    train_num = train_runs  # Use parameter
    lim = 1e-4
    itr_MAX = 1000
    thr = 0.03
    savedir = 'results'
    
    # Create results directory
    os.makedirs(savedir, exist_ok=True)
    
    # Load likelihood data
    try:
        print("Loading likelihood data...")
        likelihood_data = load_likelihood_results("lkh_win_5.npz")
        
        # Convert likelihood data to the expected format for calc_segments
        lkh_ses = []
        for session_key in sorted(likelihood_data.keys()):
            session_lkh = likelihood_data[session_key]
            # Extract the actual dictionary from the 0-dimensional numpy array
            if hasattr(session_lkh, 'item'):
                session_dict = session_lkh.item()
                if isinstance(session_dict, dict):
                    lkh_ses.append(session_dict)
                else:
                    lkh_ses.append({0: session_dict})
            else:
                lkh_ses.append({0: session_lkh})
                
    except:
        print("Warning: lkh_win_5.npz not found. Calculating likelihood first...")
        main_likelihood_calculation()
        likelihood_data = load_likelihood_results("lkh_win_5.npz")
        
        # Convert likelihood data to the expected format for calc_segments
        lkh_ses = []
        for session_key in sorted(likelihood_data.keys()):
            session_lkh = likelihood_data[session_key]
            # Extract the actual dictionary from the 0-dimensional numpy array
            if hasattr(session_lkh, 'item'):
                session_dict = session_lkh.item()
                if isinstance(session_dict, dict):
                    lkh_ses.append(session_dict)
                else:
                    lkh_ses.append({0: session_dict})
            else:
                lkh_ses.append({0: session_lkh})
    
    # Load gaze data
    try:
        loader = GazeDataLoader()
        gaze_data = loader.load_gazedata()
        semantic_attributes = loader.semantic_attributes
        semantic_attribute_seq = loader.semantic_attribute_seq
    except:
        print("Warning: gazedata.mat not found. Creating sample data...")
        gaze_data = create_sample_data()
        semantic_attributes = gaze_data['semantic_attributes']
        semantic_attribute_seq = gaze_data['semantic_attribute_seq']
    
    S = len(semantic_attribute_seq)
    print(f"Processing {S} sessions...")
    
    # Process parameters
    N_value = np.max(np.vstack(semantic_attribute_seq), axis=1)
    
    # Try to load saved sessions or create them
    try:
        print("Loading saved sessions...")
        session_data = np.load('segsessions.npz', allow_pickle=True)
        segsession = session_data['segsession'].tolist()
        session_all = session_data['session_all'].tolist()
        seglabel = session_data['seglabel'].tolist()
        seqmask = session_data['seqmask'].tolist()
    except FileNotFoundError:
        # Try loading individual session files
        try:
            print("Loading individual session files...")
            segsession = []
            session_all = []
            seglabel = []
            seqmask = []
            
            for s in range(S):
                session_file = f'session_{s}.npz'
                if os.path.exists(session_file):
                    data = np.load(session_file, allow_pickle=True)
                    segsession.append(data['segsession'])
                    session_all.append(data['session_all'])
                    seglabel.append(data['seglabel'])
                    seqmask.append(data['seqmask'])
                else:
                    raise FileNotFoundError(f"Session file {session_file} not found")
            print(f"Loaded {len(segsession)} sessions from individual files")
        except:
            print("Creating sessions and segments...")
        
        # Create session_all
        processor = SessionProcessor()
        session_all = processor.create_session_all(semantic_attribute_seq, N_value)
        
        # Calculate segments
        seg_calc = SegmentCalculator(win_size, thr)
        seglabel, seqmask = seg_calc.calc_segments(
            lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr
        )
        
        # Create segmented sessions
        segsession, seglabel = processor.create_segsessions(session_all, seglabel, seqmask)
        
        # Debug: Print shapes before saving
        print("Debug - segsession shapes:")
        for i, seg in enumerate(segsession):
            print(f"  Session {i}: {seg.shape if hasattr(seg, 'shape') else type(seg)}")
        
        # Clean up seqmask (remove empty columns)
        for s in range(S):
            if len(seqmask) > s and seqmask[s].size > 0:
                non_empty_cols = np.sum(seqmask[s], axis=0) > 0
                seqmask[s] = seqmask[s][:, non_empty_cols]
        
        # Save sessions with proper handling of object arrays
        print("Saving sessions...")
        try:
            # Save each component separately to avoid shape issues
            session_dict = {
                'segsession': segsession,
                'session_all': session_all,
                'seglabel': seglabel,
                'seqmask': seqmask,
                'win_size': win_size,
                'thr': thr
            }
            
            # Convert to object arrays for numpy
            for key, value in session_dict.items():
                if isinstance(value, list) and key in ['segsession', 'session_all', 'seglabel', 'seqmask']:
                    session_dict[key] = np.array(value, dtype=object)
            
            np.savez('segsessions.npz', **session_dict, allow_pickle=True)
            print("Sessions saved to segsessions.npz")
        except Exception as e:
            print(f"Error saving sessions: {e}")
            # Save individual components if bulk save fails
            for i, (seg, all_data, label, mask) in enumerate(zip(segsession, session_all, seglabel, seqmask)):
                np.savez(f'session_{i}.npz', 
                        segsession=seg, session_all=all_data, 
                        seglabel=label, seqmask=mask, allow_pickle=True)
            print("Sessions saved individually")
    
    # Learn topics for different numbers of topics (1 to 5)
    for topic in range(1, 6):
        print(f"\nLearning with {topic} topics...")
        
        # Train on masked/segmented data (specific aspects)
        print(f"Training on segmented data (topic={topic})...")
        params_spe, L_spe = train_multiple_runs(
            seqmask, topic, None, lim, train_num, itr_MAX
        )
        
        # Save results for specific aspects
        np.savez(f'./{savedir}/p_spe_topic{topic}.npz', 
                 p=params_spe, L=L_spe)
        
        # Save vocabulary distributions as CSV (matching MATLAB format)
        if 'vz' in params_spe:
            vz_matrix = np.hstack([v.reshape(-1, 1) for v in params_spe['vz']])
            np.savetxt(f'./{savedir}/Akq_spe_topic{topic}.csv', vz_matrix, delimiter=',')
        
        # Train on all data (general aspects)
        print(f"Training on all session data (topic={topic})...")
        params_all, L_all = train_multiple_runs(
            session_all, topic, None, lim, train_num, itr_MAX
        )
        
        # Save results for all aspects
        np.savez(f'./{savedir}/p_all_topic{topic}.npz', 
                 p=params_all, L=L_all)
        
        # Save vocabulary distributions as CSV
        if 'vz' in params_all:
            vz_matrix = np.hstack([v.reshape(-1, 1) for v in params_all['vz']])
            np.savetxt(f'./{savedir}/Akq_all_topic{topic}.csv', vz_matrix, delimiter=',')
        
        print(f"Topic {topic} completed. Results saved to {savedir}/")
    
    elapsed_time = time.time() - start_time
    print(f"\nFour-layer aspect learning completed in {elapsed_time:.2f} seconds")


def main_likelihood_calculation():
    """
    Main function for likelihood calculation (prerequisite for aspect learning)
    """
    print("Calculating likelihood values...")
    
    # Load data
    loader = GazeDataLoader()
    try:
        gaze_data = loader.load_gazedata()
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Warning: gazedata.mat not found. Creating sample data for testing.")
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
    save_likelihood_results(likelihood_results, "lkh_win_5.npz")
    print("Likelihood calculation completed and saved to lkh_win_5.npz")


def create_sample_data() -> Dict[str, Any]:
    """
    Create sample data for testing when real data is not available
    """
    print("Creating sample data for four-layer model...")
    
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
        # For semantic_attributes: (timepoints x attributes) format
        semantic_attributes.append(session_attrs.T)  
        # For semantic_attribute_seq: (attributes x timepoints) format
        semantic_attribute_seq.append(session_attrs)
    
    print(f"Created {n_sessions} sessions with {n_attributes} attributes each")
    print(f"Session data shape: {semantic_attributes[0].shape} (timepoints x attributes)")
    print(f"Sequence data shape: {semantic_attribute_seq[0].shape} (attributes x timepoints)")
    
    return {
        'semantic_attributes': semantic_attributes,
        'semantic_attribute_seq': semantic_attribute_seq
    }


def analyze_results(savedir: str = 'results'):
    """
    Analyze and display results from the four-layer model
    """
    print("\nAnalyzing Four-Layer pLSA Results...")
    
    for topic in range(1, 6):
        try:
            # Load specific aspect results
            spe_data = np.load(f'{savedir}/p_spe_topic{topic}.npz', allow_pickle=True)
            all_data = np.load(f'{savedir}/p_all_topic{topic}.npz', allow_pickle=True)
            
            print(f"\nTopic {topic}:")
            print(f"  Specific aspects - Final likelihood: {spe_data['L'][-1]:.4f}")
            print(f"  All aspects - Final likelihood: {all_data['L'][-1]:.4f}")
            print(f"  Convergence iterations (specific): {len(spe_data['L'])}")
            print(f"  Convergence iterations (all): {len(all_data['L'])}")
            
        except FileNotFoundError:
            print(f"Results for topic {topic} not found.")


def run_full_four_layer_pipeline():
    """
    Run the complete four-layer AspectEstimation pipeline
    """
    print("=" * 60)
    print("Four-Layer AspectEstimation Python Implementation")
    print("Session -> Segment -> Topic -> Vocabulary")
    print("=" * 60)
    
    # Step 1: Calculate likelihood (if needed)
    print("\nStep 1: Ensuring likelihood data is available...")
    try:
        load_likelihood_results("lkh_win_5.npz")
        print("Likelihood data found.")
    except:
        main_likelihood_calculation()
    
    # Step 2: Learn aspects using four-layer model
    print("\nStep 2: Learning aspects with four-layer pLSA...")
    main_learn_aspects_from_mask()
    
    # Step 3: Analyze results
    print("\nStep 3: Analyzing results...")
    analyze_results()
    
    print("\nFour-layer pipeline completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Four-Layer AspectEstimation Python Implementation")
    parser.add_argument("--mode", choices=["likelihood", "aspects", "full", "analyze"], 
                       default="full", help="Execution mode")
    parser.add_argument("--topics", type=int, default=5, 
                       help="Maximum number of topics to train")
    parser.add_argument("--train-runs", type=int, default=10,
                       help="Number of training runs per topic")
    
    args = parser.parse_args()
    
    if args.mode == "likelihood":
        main_likelihood_calculation()
    elif args.mode == "aspects":
        main_learn_aspects_from_mask(args.train_runs)
    elif args.mode == "analyze":
        analyze_results()
    else:
        run_full_four_layer_pipeline()
