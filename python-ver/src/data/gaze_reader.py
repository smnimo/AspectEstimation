"""
Data reader implementation for AspectEstimation
Translates ReadwithTimestamp.m and LoadGaze.m functionality
"""

import numpy as np
import os
from typing import Dict, Any


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
            os.path.join(self.base_path, "sample_data", knowledge_file),         # ./sample_data/knowledge.csv
            os.path.join(self.base_path, "..", "sample_data", knowledge_file),   # ../sample_data/knowledge.csv
            os.path.join(self.base_path, "../..", "sample_data", knowledge_file) # ../../sample_data/knowledge.csv
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
        
        raise FileNotFoundError(
            f"knowledge.csv not found; tried {[os.path.abspath(p) for p in possible_paths]}")
    
    def _convert_to_onehot(self, semantic_seq: np.ndarray) -> np.ndarray:
        """
        Convert semantic attribute sequence to one-hot encoding
        
        Args:
            semantic_seq: 5×T array where each value is 1, 2, or 3
            
        Returns:
            15×T array with one-hot encoding (5 attributes × 3 values each)
        """
        n_attributes, n_timepoints = semantic_seq.shape
        n_values = 3  # Each attribute can have values 1, 2, 3
        
        # Create one-hot encoded array: 15×T (5 attributes × 3 values)
        onehot = np.zeros((n_attributes * n_values, n_timepoints), dtype=float)
        
        for attr in range(n_attributes):
            for t in range(n_timepoints):
                value = semantic_seq[attr, t]
                if 1 <= value <= 3:
                    # Convert to one-hot index: attribute * 3 + (value - 1)
                    onehot_idx = attr * n_values + (value - 1)
                    onehot[onehot_idx, t] = 1.0
        
        return onehot
    
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
        
        gaze_data = self.load_gaze_data(csv_file)
        layout_data = self.load_item_layout(list_file)
            
        # Process the session
        session = self._process_session_data(gaze_data, layout_data, user_id, task_id)
        
        # Store in sessions dictionary
        session_key = (user_id, task_id)
        self.sessions[session_key] = session
        
        print(f"✅ Session {session_key} loaded successfully")
        print(f"   Grid positions processed: {len(session['f'])} timepoints")
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
    
    def gaze_to_items(self, gaze_data: np.ndarray) -> np.ndarray:
        """
        Convert gaze coordinates to grid positions (1-12) using the exact LoadGaze.m algorithm
        
        Args:
            gaze_data: Gaze tracking data (timepoints × 5: timestamp, right_x, right_y, left_x, left_y)
            
        Returns:
            Grid position sequence (timepoints,) with positions 1-12 or -1 for invalid
            
        Exact translation of LoadGaze.m functionality
        """
        timestamp = gaze_data[:, 0]
        gaze = gaze_data[:, 1:]
        
        # Extract x and y coordinates (columns 1,3 for x and 2,4 for y)
        x = gaze[:, [0, 2]]  # right_x, left_x
        y = gaze[:, [1, 3]]  # right_y, left_y
        
        # Handle missing data (-1) by using the other eye (both directions)
        # NOTE: intentional difference from LoadGaze.m, whose second line
        #   x(x(:, 1)==-1, 2) = x(x(:, 1)==-1, 1)
        # runs after column 1 has been filled and therefore never fills a missing column 2 (left eye):
        # such samples become (x_R - 1) / 2 < 0 and are dropped in MATLAB. Samples with both eyes
        # missing stay -1 and are invalid in both versions.
        for v in (x, y):
            right_missing = v[:, 0] == -1
            left_missing = v[:, 1] == -1
            v[right_missing, 0] = v[right_missing, 1]
            v[left_missing, 1] = v[left_missing, 0]

        # Average the two eyes
        x = np.mean(x, axis=1)
        y = np.mean(y, axis=1)

        # Grid parameters: 3 rows × 4 columns
        vN = 3  # vertical (rows)
        cN = 4  # horizontal (columns)

        # Convert normalized coordinates to grid positions
        # x coordinate: 1/4 intervals, y coordinate: 1/3 intervals
        # x == 1 or y == 1 is valid (only > 1 is rejected below) and belongs to the last column / row;
        # LoadGaze.m maps it to a non-existent cell (next row, or a target > 12)
        row = np.minimum(np.floor(y * vN), vN - 1)
        col = np.minimum(np.floor(x * cN), cN - 1)
        target_seq = row * cN + col + 1
        
        # Mark invalid coordinates (outside 0-1 range) as -1
        invalid_mask = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        target_seq[invalid_mask] = -1
        
        return target_seq.astype(int)
    
    def process_target_duration(self, gaze_data: np.ndarray, target_seq: np.ndarray, 
                               threshold_ms: float = 150.0) -> np.ndarray:
        """
        Process target sequence with duration filtering
        Exact translation of the duration processing in LoadGaze.m
        
        Args:
            gaze_data: Original gaze data with timestamps
            target_seq: Target sequence from gaze_to_items
            threshold_ms: Minimum duration threshold in milliseconds (default: 150ms)
            
        Returns:
            Filtered duration-based sequence [target_id, duration]
        """
        timestamp = gaze_data[:, 0].copy()
        
        # Normalize timestamp and convert to 100ms units
        timestamp = timestamp - timestamp[0]
        timestamp = timestamp / 1000 / 100  # Convert to 100ms units
        
        # Remove invalid targets
        valid_mask = target_seq >= 0
        timestamp = timestamp[valid_mask]
        target_seq = target_seq[valid_mask]
        
        if len(target_seq) == 0:
            return np.array([]).reshape(0, 2)
        
        # Find segments where target changes
        target_changes = np.diff(np.concatenate([[0], target_seq])) != 0
        change_indices = np.where(target_changes)[0]
        
        if len(change_indices) == 0:
            return np.array([]).reshape(0, 2)
        
        # Create target_tb_te: [target, start_time, end_time]
        target_tb_te = []
        for i in range(len(change_indices)):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else len(target_seq)
            
            target_id = target_seq[start_idx]
            start_time = timestamp[start_idx]
            end_time = timestamp[end_idx - 1] if end_idx > start_idx else start_time
            
            target_tb_te.append([target_id, start_time, end_time])
        
        if len(target_tb_te) == 0:
            return np.array([]).reshape(0, 2)
            
        target_tb_te = np.array(target_tb_te)
        
        # Calculate durations
        target_dur = target_tb_te[:, 2] - target_tb_te[:, 1]
        
        # Apply threshold filter (150ms = 1.5 in 100ms units)
        threshold_units = threshold_ms / 100.0
        valid_duration = target_dur >= threshold_units
        target_tb_te = target_tb_te[valid_duration]
        
        if len(target_tb_te) == 0:
            return np.array([]).reshape(0, 2)
        
        # Merge consecutive identical targets
        T = len(target_tb_te)
        t = 1
        while t < T:
            if target_tb_te[t, 0] == target_tb_te[t-1, 0]:
                # Merge: extend end time of previous segment
                target_tb_te[t-1, 2] = target_tb_te[t, 2]
                # Remove current segment
                target_tb_te = np.delete(target_tb_te, t, axis=0)
                T -= 1
            else:
                t += 1
        
        # Final result: [target_id, duration]
        final_durations = target_tb_te[:, 2] - target_tb_te[:, 1]
        target_dur_result = np.column_stack([target_tb_te[:, 0], final_durations])
        
        return target_dur_result
    
    def _process_session_data(self, gaze_data: np.ndarray, layout_data: np.ndarray,
                             user_id: int, task_id: int) -> dict:
        """
        Process raw gaze and layout data into semantic attributes
        Exact translation of LoadGaze.m + ReadwithTimestamp.m processing
        
        Args:
            gaze_data: Raw gaze tracking data (timepoints × 5)
            layout_data: Item layout data from .list file
            user_id: User identifier
            task_id: Task identifier
            
        Returns:
            Processed session data dictionary
        """
        # Load gaze sequence using LoadGaze.m equivalent
        # First get grid positions (1-12), then apply duration processing
        grid_positions = self.gaze_to_items(gaze_data)
        target_duration = self.process_target_duration(gaze_data, grid_positions)
        
        # Extract task from first item (as in ReadwithTimestamp.m)
        task = layout_data[0] // 100  # floor(cont(1)/100)
        
        # Create item list by removing task prefix
        item_list = layout_data - task * 100  # list = cont - task * 100
        
        # Use filtered sequence from LoadGaze (target_duration contains [target_id, duration])
        if len(target_duration) > 0:
            f = target_duration[:, 0].astype(int)  # f = seq(:, 1)
            dur = target_duration[:, 1]            # dur = seq(:, 2)
        else:
            f = np.array([], dtype=int)
            dur = np.array([])
        
        # Ensure knowledge base is loaded
        if self.knowledge_base is None:
            self.load_knowledge_base()
        
        # Create semantic_attribute_seq as in MATLAB: KB(list(f), :)'
        # This is the key mapping: f indexes into item_list, then into KB
        # (MATLAB errors on an invalid index; zero-filling would create attribute value 0)
        if np.any((item_list < 1) | (item_list > self.knowledge_base.shape[0])):
            raise ValueError(f"item ids out of the knowledge base range: {layout_data}")
        if len(f) > 0:
            if np.any((f < 1) | (f > len(item_list))):
                raise ValueError(f"grid positions out of 1..{len(item_list)}: {np.unique(f)}")
            semantic_seq = self.knowledge_base[item_list[f - 1] - 1, :].T

            # Convert to one-hot encoding for feature values
            # Each attribute can have values 1, 2, 3 -> 15 features total (5 attributes × 3 values)
            semantic_seq_onehot = self._convert_to_onehot(semantic_seq)
        else:
            semantic_seq = np.zeros((self.knowledge_base.shape[1], 0), dtype=int)
            semantic_seq_onehot = np.zeros((15, 0), dtype=float)  # 5 attributes × 3 values
        
        # Create semantic_attributes: KB(list, :)
        semantic_attributes = np.zeros((len(item_list), self.knowledge_base.shape[1]), dtype=int)
        for i, item in enumerate(item_list):
            kb_idx = item - 1  # Convert to 0-based
            if 0 <= kb_idx < self.knowledge_base.shape[0]:
                semantic_attributes[i, :] = self.knowledge_base[kb_idx, :]
        
        # Add candidate information based on task (as in ReadwithTimestamp.m)
        if task == 1:
            candidate = [1, 2]
        elif task == 2:
            candidate = [8, 11] 
        elif task == 3:
            candidate = [3, 4]
        else:
            candidate = []
        
        # Calculate total duration in seconds (as in MATLAB)
        total_duration = (gaze_data[-1, 0] - gaze_data[0, 0]) / 1000000  # ms to seconds
        
        session_data = {
            'f': f,                                    # Filtered item sequence
            'dur': dur,                               # Duration sequence  
            'list': item_list,                        # Item list (task prefix removed)
            'task': task,                             # Task number
            'usr': user_id,                           # User ID
            'totalduration': total_duration,          # Total session duration
            'semantic_attribute_seq': semantic_seq,   # KB(list(f), :)' - original 5×T
            'semantic_attribute_seq_onehot': semantic_seq_onehot,  # One-hot encoded 15×T
            'semantic_attributes': semantic_attributes, # KB(list, :)
            'candidate': candidate,                   # Task-specific candidates
            'raw_gaze': gaze_data,                    # Original gaze data
            'layout_data': layout_data,               # Original layout data
            'target_duration': target_duration,       # Duration-filtered sequence
            'n_timepoints': len(f)                    # Number of timepoints
        }
        
        return session_data
    
    def process_session(self, csv_file: str, list_file: str, user_id: int, task_id: int = None) -> Dict[str, Any]:
        """
        Process a single session (gaze data + layout) without printing (LoadGaze.m + ReadwithTimestamp.m)

        Args:
            csv_file: Gaze data file
            list_file: Item layout file
            user_id: User identifier
            task_id: Ignored; the task is taken from the .list file (floor(cont(1) / 100)) as in MATLAB

        Returns:
            Processed session data (see _process_session_data)
        """
        if self.knowledge_base is None:
            self.load_knowledge_base()
        layout = np.loadtxt(list_file, dtype=int)
        return self._process_session_data(np.loadtxt(csv_file, delimiter=','), layout, user_id, int(layout[0] // 100))
    
    def load_all_sessions(self, timestamp_list: str, experiment_dir: str = None) -> Dict[str, Any]:
        """
        Load all sessions listed in timestamp.list (main loop of ReadwithTimestamp.m)

        Every 4 consecutive lines belong to one user; the 4th session of each user is skipped
        (session_id = mod(count, 4) + 1, new user when session_id == 1, skip session_id == 4).
        The task is taken from the .list file (floor(cont(1) / 100)). Blank lines are ignored
        (MATLAB would fail on them).

        Args:
            timestamp_list: Path to timestamp.list (one session name per line)
            experiment_dir: Directory with <name>.csv / <name>.list (default: directory of timestamp_list)

        Returns:
            {'session', 'semantic_attribute_seq', 'semantic_attributes'}: dicts keyed by (usr, task),
            'KB', and 'n_usr'; use main.from_reader_sessions for MATLAB's cell(:) order

        Raises:
            FileNotFoundError / ValueError: missing or invalid files (no fallback to fake data)
        """
        if self.knowledge_base is None:
            self.load_knowledge_base()
        if experiment_dir is None:
            experiment_dir = os.path.dirname(os.path.abspath(timestamp_list))
        with open(timestamp_list, 'r') as fp:
            names = [line.strip() for line in fp if line.strip()]

        sessions = {}
        usr = 0
        for count, name in enumerate(names):
            session_id = count % 4 + 1
            if session_id == 1:
                usr += 1
            if session_id == 4:
                continue
            session = self.process_session(os.path.join(experiment_dir, f"{name}.csv"),
                                           os.path.join(experiment_dir, f"{name}.list"), usr)
            key = (usr, int(session['task']))
            if key in sessions:
                print(f"Warning: session {key} appears twice; {name} overwrites it (as in MATLAB)")
            session['name'] = name
            sessions[key] = session

        self.sessions = sessions
        self.semantic_attribute_seq = {k: s['semantic_attribute_seq'] for k, s in sessions.items()}
        self.semantic_attributes = {k: s['semantic_attributes'] for k, s in sessions.items()}
        return {
            'session': sessions,
            'semantic_attribute_seq': self.semantic_attribute_seq,
            'semantic_attributes': self.semantic_attributes,
            'KB': self.knowledge_base,
            'n_usr': usr
        }

    def save_gazedata(self, data: Dict[str, Any], output_file: str = "gazedata.mat"):
        """
        Save sessions as a MATLAB gazedata.mat readable by main.load_gazedata_mat

        semantic_attribute_seq / semantic_attributes are saved as n_usr x n_task cell arrays
        indexed {usr, task}, as in ReadwithTimestamp.m, together with KB. The session structs
        (f, dur, list, ...) are not saved.

        Args:
            data: Output of load_all_sessions
            output_file: Output .mat path

        Translates: save('gazedata', 'session', 'semantic_attribute_seq', 'semantic_attributes', 'KB');
        """
        import scipy.io
        keys = list(data['session'].keys())
        n_usr = max(k[0] for k in keys)
        n_task = max(k[1] for k in keys)
        missing = [(u, t) for u in range(1, n_usr + 1) for t in range(1, n_task + 1) if (u, t) not in data['session']]
        if missing:
            print(f"Warning: empty cells for (usr, task) = {missing}")
        cells = {}
        for name in ('semantic_attribute_seq', 'semantic_attributes'):
            c = np.empty((n_usr, n_task), dtype=object)
            for u in range(n_usr):
                for t in range(n_task):
                    c[u, t] = np.asarray(data[name].get((u + 1, t + 1), np.zeros((0, 0))), dtype=float)
            cells[name] = c
        scipy.io.savemat(output_file, {**cells, 'KB': np.asarray(data['KB'], dtype=float)})
        print(f"Saved {len(keys)} sessions ({n_usr} users x {n_task} tasks) to {output_file}")


def main_read_gaze_data():
    """
    Read all sessions and save gazedata.mat (ReadwithTimestamp.m)

    Usage (from python-ver/):
      python src/data/gaze_reader.py --timestamp-list ../experiment/timestamp.list \
          --knowledge ../knowledge/knowledge.csv --out gazedata.mat
      python main.py --mat gazedata.mat --eval
    """
    import argparse
    parser = argparse.ArgumentParser(description="Read gaze data (ReadwithTimestamp.m)")
    parser.add_argument('--timestamp-list', required=True, help="timestamp.list (one session name per line)")
    parser.add_argument('--experiment-dir', default=None,
                        help="directory with <name>.csv / .list (default: directory of timestamp.list)")
    parser.add_argument('--knowledge', required=True, help="knowledge.csv (KB, 12 items x 5 attributes)")
    parser.add_argument('--out', default='gazedata.mat')
    args = parser.parse_args()

    reader = GazeDataReader(base_path='.', knowledge_csv_path=args.knowledge)
    reader.load_knowledge_base()
    data = reader.load_all_sessions(args.timestamp_list, args.experiment_dir)
    reader.save_gazedata(data, args.out)
    return data


if __name__ == "__main__":
    main_read_gaze_data()
