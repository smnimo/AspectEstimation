"""
Data reader implementation for AspectEstimation
Translates ReadwithTimestamp.m and LoadGaze.m functionality
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Any
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import normalize


class GazeDataReader:
    """
    Main data reader class that replicates ReadwithTimestamp.m functionality
    
    Handles:
    1. Knowledge base loading (knowledge.csv) - Feature→Observation mapping
    2. Gaze data loading (.csv files) - Raw eye tracking data
    3. Item layout loading (.list files) - Item arrangement per session
    4. Conversion from gaze coordinates to semantic attributes
    """
    
    def __init__(self, base_path: str = "../../..", knowledge_csv_path: str = None):
        """
        Initialize the gaze data reader
        
        Args:
            base_path: Base directory containing data files (default: relative to AspectEstimation root)
            knowledge_csv_path: Explicit path to knowledge.csv file (optional)
        """
        self.base_path = base_path
        self.knowledge_csv_path = knowledge_csv_path
        self.knowledge_base = None  # KB matrix from knowledge.csv
        self.sessions = {}          # Processed session data
        self.semantic_attribute_seq = {}  # Sequence of attributes per session
        self.semantic_attributes = {}     # All attributes per session
        
    def load_knowledge_base(self, knowledge_file: str = None) -> np.ndarray:
        """
        Load the knowledge base (Feature→Observation mapping)
        
        Args:
            knowledge_file: Path to knowledge.csv (optional, uses explicit path if set)
            
        Returns:
            Knowledge base matrix (items × attributes)
            
        Translates: KB = csvread('../knowledge/knowledge.csv');
        """
        # Use explicit path if provided during initialization
        if self.knowledge_csv_path:
            knowledge_path = self.knowledge_csv_path
            if os.path.exists(knowledge_path):
                try:
                    self.knowledge_base = np.loadtxt(knowledge_path, delimiter=',', dtype=int)
                    print(f"✅ Loaded knowledge base from: {knowledge_path}")
                    print(f"Shape: {self.knowledge_base.shape} (items × attributes)")
                    return self.knowledge_base
                except Exception as e:
                    print(f"Error loading {knowledge_path}: {e}")
        
        # Fallback to relative path search
        if knowledge_file is None:
            knowledge_file = "knowledge.csv"
            
        # Try multiple possible paths
        possible_paths = [
            os.path.join(self.base_path, knowledge_file),           # ./knowledge.csv
            os.path.join(self.base_path, "..", knowledge_file),     # ../knowledge.csv
            os.path.join(self.base_path, "../..", knowledge_file),  # ../../knowledge.csv
        ]
        
        for knowledge_path in possible_paths:
            if os.path.exists(knowledge_path):
                try:
                    self.knowledge_base = np.loadtxt(knowledge_path, delimiter=',', dtype=int)
                    print(f"✅ Loaded knowledge base from: {knowledge_path}")
                    print(f"Shape: {self.knowledge_base.shape} (items × attributes)")
                    return self.knowledge_base
                except Exception as e:
                    print(f"Error loading {knowledge_path}: {e}")
                    continue
        
        print(f"Warning: knowledge.csv not found in any expected location.")
        print(f"Tried paths: {[os.path.abspath(p) for p in possible_paths]}")
        print("Using sample knowledge base.")
        # Create sample knowledge base (12 items × 5 attributes)
        self.knowledge_base = self._create_sample_knowledge_base()
        return self.knowledge_base
    
    def _create_sample_knowledge_base(self) -> np.ndarray:
        """Create a sample knowledge base for testing"""
        # Sample 12 items × 5 attributes with values 1-3
        kb = np.array([
            [2, 3, 3, 3, 2],
            [2, 2, 3, 3, 3],
            [1, 2, 2, 3, 3],
            [1, 3, 1, 3, 1],
            [3, 3, 3, 2, 3],
            [3, 2, 2, 2, 2],
            [3, 1, 1, 2, 3],
            [2, 1, 1, 2, 1],
            [3, 2, 3, 1, 1],
            [1, 1, 2, 1, 2],
            [2, 1, 2, 1, 1],
            [1, 3, 1, 1, 2]
        ])
        print(f"Created sample knowledge base: {kb.shape}")
        return kb
    
    def load_session_from_files(self, csv_file: str, list_file: str, 
                               user_id: int = 1, task_id: int = 1) -> dict:
        """
        Load a single session from explicit CSV and LIST file paths
        
        Args:
            csv_file: Path to gaze data CSV file (e.g., "170116024150092.csv")
            list_file: Path to item layout LIST file (e.g., "170116024150092.list")
            user_id: User ID for this session
            task_id: Task ID for this session
            
        Returns:
            Session data dictionary with processed gaze and semantic data
        """
        print(f"Loading session from files:")
        print(f"  CSV: {csv_file}")
        print(f"  LIST: {list_file}")
        print(f"  User: {user_id}, Task: {task_id}")
        
        # Load gaze data
        try:
            gaze_data = self.load_gaze_data(csv_file)
        except Exception as e:
            print(f"Error loading gaze data: {e}")
            return None
            
        # Load layout data
        try:
            layout_data = self.load_item_layout(list_file)
        except Exception as e:
            print(f"Error loading layout data: {e}")
            return None
            
        # Process the session
        session = self._process_session_data(gaze_data, layout_data, user_id, task_id)
        
        # Store in sessions dictionary
        session_key = (user_id, task_id)
        self.sessions[session_key] = session
        
        print(f"✅ Session {session_key} loaded successfully")
        print(f"   Item sequence length: {len(session['item_sequence'])}")
        print(f"   Semantic sequence shape: {session['semantic_attribute_seq'].shape}")
        
        return session
    
    def load_gaze_data(self, csv_file: str) -> np.ndarray:
        """
        Load gaze tracking data from CSV file
        
        Args:
            csv_file: Path to gaze data CSV file
            
        Returns:
            Gaze data array (timepoints × 5: timestamp, right_x, right_y, left_x, left_y)
            
        Translates: data = csvread(strcat(h, tline, '.csv'));
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Gaze data file not found: {csv_file}")
            
        data = np.loadtxt(csv_file, delimiter=',')
        print(f"Loaded gaze data: {data.shape} (timepoints × 5)")
        return data
    
    def load_item_layout(self, list_file: str) -> np.ndarray:
        """
        Load item layout information from .list file
        
        Args:
            list_file: Path to layout .list file
            
        Returns:
            Item layout array
            
        Translates: cont = csvread(strcat(h, tline, '.list'));
        """
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Layout file not found: {list_file}")
            
        layout = np.loadtxt(list_file, dtype=int)
        print(f"Loaded item layout: {layout.shape}")
        return layout
    
    def gaze_to_items(self, gaze_data: np.ndarray, grid_size: Tuple[int, int] = (3, 4)) -> np.ndarray:
        """
        Convert gaze coordinates to item indices
        
        Args:
            gaze_data: Gaze tracking data (timepoints × 5)
            grid_size: Grid layout (rows, cols) - default 3×4
            
        Returns:
            Item sequence (timepoints,) with item indices (1-12) or 0 for invalid
            
        Approximates LoadGaze.m functionality for item identification
        """
        rows, cols = grid_size
        n_items = rows * cols
        
        # Extract gaze coordinates (use right eye, fallback to left eye)
        x_coords = np.where(gaze_data[:, 1] != -1, gaze_data[:, 1], gaze_data[:, 3])
        y_coords = np.where(gaze_data[:, 2] != -1, gaze_data[:, 2], gaze_data[:, 4])
        
        # Convert normalized coordinates to grid positions
        item_sequence = np.zeros(len(gaze_data), dtype=int)
        
        for t in range(len(gaze_data)):
            if x_coords[t] != -1 and y_coords[t] != -1:
                # Convert normalized coordinates (0-1) to grid indices
                col = int(x_coords[t] * cols)
                row = int(y_coords[t] * rows)
                
                # Ensure within bounds
                col = max(0, min(col, cols - 1))
                row = max(0, min(row, rows - 1))
                
                # Convert to item index (1-based)
                item_sequence[t] = row * cols + col + 1
            else:
                item_sequence[t] = 0  # Invalid gaze
        
        return item_sequence
    
    def _process_session_data(self, gaze_data: np.ndarray, layout_data: np.ndarray,
                             user_id: int, task_id: int) -> dict:
        """
        Process raw gaze and layout data into semantic attributes
        
        Args:
            gaze_data: Raw gaze tracking data (timepoints × 5)
            layout_data: Item layout data
            user_id: User identifier
            task_id: Task identifier
            
        Returns:
            Processed session data dictionary
        """
        # Convert gaze coordinates to item sequence
        item_sequence = self.gaze_to_items(gaze_data)
        
        # Ensure knowledge base is loaded
        if self.knowledge_base is None:
            self.load_knowledge_base()
        
        # Convert item sequence to semantic attributes
        semantic_seq = self.items_to_semantic_attributes(item_sequence)
        
        session_data = {
            'user_id': user_id,
            'task_id': task_id,
            'raw_gaze': gaze_data,
            'layout_data': layout_data,
            'item_sequence': item_sequence,
            'semantic_attribute_seq': semantic_seq,
            'n_timepoints': len(item_sequence)
        }
        
        return session_data
    
    def process_session(self, csv_file: str, list_file: str, user_id: int, task_id: int) -> Dict[str, Any]:
        """
        Process a single session (gaze data + layout)
        
        Args:
            csv_file: Gaze data file
            list_file: Item layout file
            user_id: User identifier
            task_id: Task identifier
            
        Returns:
            Processed session data
            
        Translates the main processing loop in ReadwithTimestamp.m
        """
        # Load data
        gaze_data = self.load_gaze_data(csv_file)
        item_layout = self.load_item_layout(list_file)
        
        # Convert gaze to item sequence
        item_sequence = self.gaze_to_items(gaze_data)
        
        # Calculate task and get actual item list
        task = task_id
        item_list = item_layout - task * 100  # Remove task prefix
        
        # Convert item sequence to attribute sequence using knowledge base
        # semantic_attribute_seq{usr, task} = KB(list(f), :)';
        valid_timepoints = item_sequence > 0
        semantic_seq = np.zeros((self.knowledge_base.shape[1], len(item_sequence)), dtype=int)
        
        for t in range(len(item_sequence)):
            if valid_timepoints[t]:
                item_idx = item_sequence[t] - 1  # Convert to 0-based
                if 0 <= item_idx < len(item_list):
                    knowledge_idx = item_list[item_idx] - 1  # Convert to 0-based
                    if 0 <= knowledge_idx < self.knowledge_base.shape[0]:
                        semantic_seq[:, t] = self.knowledge_base[knowledge_idx, :]
        
        # Create semantic attributes (all items in this session)
        semantic_attrs = self.knowledge_base[item_list - 1, :]  # Convert to 0-based
        
        # Session metadata
        session_data = {
            'f': item_sequence,
            'dur': np.diff(gaze_data[:, 0]) / 1000000,  # Duration in seconds
            'list': item_list,
            'task': task,
            'usr': user_id,
            'totalduration': (gaze_data[-1, 0] - gaze_data[0, 0]) / 1000000,
            'semantic_attribute_seq': semantic_seq,
            'semantic_attributes': semantic_attrs
        }
        
        return session_data
    
    def items_to_semantic_attributes(self, item_sequence: np.ndarray) -> np.ndarray:
        """
        Convert item sequence to semantic attribute sequence using knowledge base
        
        Args:
            item_sequence: Array of item indices (1-based, 0 for invalid)
            
        Returns:
            Semantic attribute sequence (features × time) using offset encoding
        """
        if self.knowledge_base is None:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")
        
        n_items, n_attributes = self.knowledge_base.shape
        n_values = np.max(self.knowledge_base, axis=0)
        total_features = np.sum(n_values)
        
        # Calculate offsets for attribute encoding
        offsets = [0]
        for i in range(len(n_values) - 1):
            offsets.append(offsets[-1] + n_values[i])
        
        # Create output matrix
        n_timepoints = len(item_sequence)
        semantic_seq = np.zeros((total_features, n_timepoints))
        
        for t in range(n_timepoints):
            item_idx = item_sequence[t]
            if item_idx > 0 and item_idx <= n_items:  # Valid item (1-based)
                # Get attributes for this item
                item_attrs = self.knowledge_base[item_idx - 1, :]  # Convert to 0-based
                
                # Encode using offsets
                for attr in range(n_attributes):
                    attr_value = item_attrs[attr]
                    if attr_value > 0:  # Valid attribute value
                        feature_idx = offsets[attr] + attr_value - 1  # Convert to 0-based
                        semantic_seq[feature_idx, t] = 1
        
        return semantic_seq
    
    def load_all_sessions(self, timestamp_list: str = None, 
                         experiment_dir: str = None) -> Dict[str, Any]:
        """
        Load all sessions from timestamp list
        
        Args:
            timestamp_list: Path to timestamp.list file (if None, looks for actual data files)
            experiment_dir: Directory containing experiment files (if None, looks for actual data)
            
        Returns:
            Complete session data structure
            
        Translates the main loop in ReadwithTimestamp.m
        """
        if self.knowledge_base is None:
            self.load_knowledge_base()
        
        # If no timestamp list provided, try to find actual data files
        if timestamp_list is None:
            return self._load_available_data_files()
        
        timestamp_path = os.path.join(self.base_path, timestamp_list)
        if not os.path.exists(timestamp_path):
            print(f"Warning: {timestamp_path} not found.")
            print("Trying to load available data files directly...")
            return self._load_available_data_files()
        
        sessions = {}
        semantic_attribute_seq = {}
        semantic_attributes = {}
        
        with open(timestamp_path, 'r') as f:
            lines = f.readlines()
        
        count = 0
        usr = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            session_id = (count % 4) + 1
            if session_id == 1:
                usr += 1
            
            if session_id != 4:  # Skip session_id == 4
                csv_file = os.path.join(self.base_path, experiment_dir or "experiment", f"{line}.csv")
                list_file = os.path.join(self.base_path, experiment_dir or "experiment", f"{line}.list")
                
                try:
                    # Extract task from first item in list
                    item_layout = self.load_item_layout(list_file)
                    task = item_layout[0] // 100
                    
                    session_data = self.process_session(csv_file, list_file, usr, task)
                    
                    sessions[(usr, task)] = session_data
                    semantic_attribute_seq[(usr, task)] = session_data['semantic_attribute_seq']
                    semantic_attributes[(usr, task)] = session_data['semantic_attributes']
                    
                    print(f"Processed session: user {usr}, task {task}")
                    
                except Exception as e:
                    print(f"Error processing {line}: {e}")
            
            count += 1
        
        return {
            'session': sessions,
            'semantic_attribute_seq': semantic_attribute_seq,
            'semantic_attributes': semantic_attributes,
            'KB': self.knowledge_base
        }
    
    def _load_available_data_files(self) -> Dict[str, Any]:
        """
        Try to load any available .csv and .list files in the directory
        """
        print("Looking for available data files...")
        
        # Look for .csv and .list files
        csv_files = []
        list_files = []
        
        # Check if there are any data files in the base directory
        if os.path.exists(self.base_path):
            for file in os.listdir(self.base_path):
                if file.endswith('.csv') and not file == 'knowledge.csv':
                    csv_files.append(file)
                elif file.endswith('.list'):
                    list_files.append(file)
        
        if not csv_files or not list_files:
            print("No data files found. Creating sample session data.")
            return self._create_sample_sessions()
        
        print(f"Found {len(csv_files)} CSV files and {len(list_files)} list files")
        
        sessions = {}
        semantic_attribute_seq = {}
        semantic_attributes = {}
        
        # Process the first available pair for demonstration
        if csv_files and list_files:
            csv_file = os.path.join(self.base_path, csv_files[0])
            list_file = os.path.join(self.base_path, list_files[0])
            
            try:
                # Extract task from first item in list
                item_layout = self.load_item_layout(list_file)
                task = item_layout[0] // 100
                usr = 1
                
                session_data = self.process_session(csv_file, list_file, usr, task)
                
                sessions[(usr, task)] = session_data
                semantic_attribute_seq[(usr, task)] = session_data['semantic_attribute_seq']
                semantic_attributes[(usr, task)] = session_data['semantic_attributes']
                
                print(f"Processed available data: user {usr}, task {task}")
                print(f"Files: {csv_files[0]}, {list_files[0]}")
                
            except Exception as e:
                print(f"Error processing available files: {e}")
                return self._create_sample_sessions()
        
        return {
            'session': sessions,
            'semantic_attribute_seq': semantic_attribute_seq,
            'semantic_attributes': semantic_attributes,
            'KB': self.knowledge_base
        }
    
    def _create_sample_sessions(self) -> Dict[str, Any]:
        """Create sample session data for testing"""
        print("Creating sample session data...")
        
        sessions = {}
        semantic_attribute_seq = {}
        semantic_attributes = {}
        
        # Create 2 users × 3 tasks sample data
        for usr in range(1, 3):  # Users 1, 2
            for task in range(1, 4):  # Tasks 1, 2, 3
                # Sample data
                n_timepoints = 100
                item_sequence = np.random.randint(1, 13, n_timepoints)  # 12 items
                
                # Create semantic sequence using knowledge base
                semantic_seq = np.zeros((self.knowledge_base.shape[1], n_timepoints), dtype=int)
                for t in range(n_timepoints):
                    item_idx = item_sequence[t] - 1
                    semantic_seq[:, t] = self.knowledge_base[item_idx, :]
                
                # Session data
                session_data = {
                    'f': item_sequence,
                    'task': task,
                    'usr': usr,
                    'semantic_attribute_seq': semantic_seq,
                    'semantic_attributes': self.knowledge_base
                }
                
                sessions[(usr, task)] = session_data
                semantic_attribute_seq[(usr, task)] = semantic_seq
                semantic_attributes[(usr, task)] = self.knowledge_base
        
        return {
            'session': sessions,
            'semantic_attribute_seq': semantic_attribute_seq,
            'semantic_attributes': semantic_attributes,
            'KB': self.knowledge_base
        }
    
    def save_gazedata(self, output_file: str = "gazedata.npz"):
        """
        Save processed data in format compatible with MATLAB gazedata.mat
        
        Args:
            output_file: Output file path
            
        Translates: save('gazedata', 'session', 'semantic_attribute_seq', 'semantic_attributes', 'KB');
        """
        data = self.load_all_sessions()
        
        # Convert to format compatible with existing Python code
        session_list = []
        semantic_seq_list = []
        semantic_attr_list = []
        
        for (usr, task), session in data['session'].items():
            session_list.append(session)
            semantic_seq_list.append(session['semantic_attribute_seq'])
            semantic_attr_list.append(session['semantic_attributes'])
        
        np.savez(output_file,
                session=np.array(session_list, dtype=object),
                semantic_attribute_seq=np.array(semantic_seq_list, dtype=object),
                semantic_attributes=np.array(semantic_attr_list, dtype=object),
                KB=data['KB'],
                allow_pickle=True)
        
        print(f"Saved processed gaze data to {output_file}")
        return data


def main_read_gaze_data():
    """
    Main function to read and process gaze data
    Replicates ReadwithTimestamp.m functionality
    """
    print("Reading gaze data (ReadwithTimestamp.m equivalent)...")
    print("Using relative paths to access actual data files...")
    
    # Initialize reader with relative path to AspectEstimation root
    reader = GazeDataReader("../../..")
    
    # Load and process all sessions
    data = reader.load_all_sessions()
    
    # Save in compatible format
    reader.save_gazedata()
    
    # Display summary
    print(f"\nProcessed data summary:")
    print(f"Knowledge base shape: {data['KB'].shape}")
    print(f"Number of sessions: {len(data['session'])}")
    print(f"Users and tasks: {list(data['session'].keys())}")
    
    return data


if __name__ == "__main__":
    main_read_gaze_data()
