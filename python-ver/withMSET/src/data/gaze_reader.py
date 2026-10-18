"""
Data reader implementation for AspectEstimation
Translates ReadwithTimestamp.m and LoadGaze.m functionality

The catalog size and the attributes are taken from knowledge.csv (data.knowledge) and the screen grid from `grid`
(rows, columns; default 3 x 4 as in the paper). A .list file holds the item ids of the grid cells in row-major
order, each plus 100 x task.
"""

import numpy as np
import os
import sys
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src (when run as a script)
from data.knowledge import Knowledge, load_knowledge  # noqa: E402


def gaze_xy(gaze_data: np.ndarray):
    """
    Gaze position averaged over both eyes (LoadGaze.m), normalized 0..1; -1 where both eyes are missing

    A missing eye (-1) is filled from the other one in both directions.
    NOTE: intentional difference from LoadGaze.m, whose second line
      x(x(:, 1)==-1, 2) = x(x(:, 1)==-1, 1)
    runs after column 1 has been filled and therefore never fills a missing column 2 (left eye):
    such samples become (x_R - 1) / 2 < 0 and are dropped in MATLAB.

    Args:
        gaze_data: (timepoints x 5) timestamp, right_x, right_y, left_x, left_y

    Returns:
        (x, y) arrays of length timepoints
    """
    x = gaze_data[:, [1, 3]].astype(float)  # right_x, left_x
    y = gaze_data[:, [2, 4]].astype(float)  # right_y, left_y
    for v in (x, y):
        right_missing = v[:, 0] == -1
        left_missing = v[:, 1] == -1
        v[right_missing, 0] = v[right_missing, 1]
        v[left_missing, 1] = v[left_missing, 0]
    return np.mean(x, axis=1), np.mean(y, axis=1)


class GazeDataReader:
    """
    Main data reader class that replicates ReadwithTimestamp.m functionality
    
    Handles:
    1. Knowledge base loading (knowledge.csv) - Feature→Observation mapping
    2. Gaze data loading (.csv files) - Raw eye tracking data
    3. Item layout loading (.list files) - Item arrangement per session
    4. Conversion from gaze coordinates to semantic attributes
    """
    
    def __init__(self, base_path: str = "../../..", knowledge_csv_path: str = None, grid: Tuple[int, int] = (3, 4)):
        """
        Initialize the gaze data reader
        
        Args:
            base_path: Base directory containing data files (default: relative to AspectEstimation root)
            knowledge_csv_path: Explicit path to knowledge.csv file (optional)
            grid: (rows, columns) of the item grid on the screen
        """
        self.base_path = base_path
        self.knowledge_csv_path = knowledge_csv_path
        self.grid = (int(grid[0]), int(grid[1]))
        self.knowledge = None       # data.knowledge.Knowledge from knowledge.csv
        self.knowledge_base = None  # KB matrix (items x attributes, values 1..N) if every item has one value per attribute
        self.sessions = {}          # Processed session data
        self.semantic_attribute_seq = {}  # Sequence of attributes per session
        self.semantic_attributes = {}     # All attributes per session
        
    def _read_knowledge(self, knowledge_path: str) -> Knowledge:
        self.knowledge = load_knowledge(knowledge_path)
        try:
            self.knowledge_base = self.knowledge.value_matrix()
        except ValueError:
            self.knowledge_base = None  # multi-valued categorical attributes
        print(f"✅ Loaded knowledge base from: {knowledge_path}")
        print(f"{self.knowledge.n_items} items x {self.knowledge.n_attributes} attributes "
              f"({self.knowledge.n_features} features)")
        return self.knowledge

    def load_knowledge_base(self, knowledge_file: str = None) -> Knowledge:
        """
        Load the knowledge base (item attributes; data.knowledge.load_knowledge)
        
        Args:
            knowledge_file: Path to knowledge.csv (optional, uses explicit path if set)
            
        Returns:
            Knowledge (also self.knowledge; self.knowledge_base is the KB matrix when it exists)
            
        Translates: KB = csvread('../knowledge/knowledge.csv');
        """
        # Use explicit path if provided during initialization
        if self.knowledge_csv_path:
            if not os.path.exists(self.knowledge_csv_path):
                raise FileNotFoundError(f"knowledge.csv not found: {self.knowledge_csv_path}")
            return self._read_knowledge(self.knowledge_csv_path)
        
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
                return self._read_knowledge(knowledge_path)
        
        raise FileNotFoundError(
            f"knowledge.csv not found; tried {[os.path.abspath(p) for p in possible_paths]}")
    
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
        print(f"   Feature sequence shape: {session['feature_seq'].shape}")
        
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
        Convert gaze coordinates to grid positions (1..rows x columns, row-major) as in LoadGaze.m
        (with the fixes noted in gaze_xy / below)
        
        Args:
            gaze_data: Gaze tracking data (timepoints × 5: timestamp, right_x, right_y, left_x, left_y)
            
        Returns:
            Grid position sequence (timepoints,) with positions 1..rows x columns or -1 for invalid
        """
        x, y = gaze_xy(gaze_data)
        
        # Grid parameters (paper: 3 rows × 4 columns)
        vN, cN = self.grid  # vertical (rows), horizontal (columns)

        # Convert normalized coordinates to grid positions
        # x coordinate: 1/cN intervals, y coordinate: 1/vN intervals
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
        Process raw gaze and layout data into item attributes
        Translation of LoadGaze.m + ReadwithTimestamp.m processing, for any knowledge.csv and grid
        
        Args:
            gaze_data: Raw gaze tracking data (timepoints × 5)
            layout_data: Item layout data from .list file
            user_id: User identifier
            task_id: Task identifier
            
        Returns:
            Processed session data dictionary
        """
        # Load gaze sequence using LoadGaze.m equivalent
        # First get grid positions (1..rows x columns), then apply duration processing
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
        if self.knowledge is None:
            self.load_knowledge_base()
        knowledge = self.knowledge
        
        # Map the browsed grid positions to items: list(f) as in MATLAB
        # (MATLAB errors on an invalid index; zero-filling would create attribute value 0)
        n_cells = self.grid[0] * self.grid[1]
        if len(item_list) != n_cells:
            raise ValueError(f"the .list has {len(item_list)} items, the {self.grid[0]} x {self.grid[1]} grid "
                             f"has {n_cells} cells: {layout_data}")
        if np.any((item_list < 1) | (item_list > knowledge.n_items)):
            raise ValueError(f"item ids out of the knowledge base range 1..{knowledge.n_items}: {layout_data}")
        if np.any((f < 1) | (f > len(item_list))):
            raise ValueError(f"grid positions out of 1..{len(item_list)}: {np.unique(f)}")
        browsed = item_list[f - 1].astype(int)
        
        # Attributes of the browsed items (feature_seq: features x T, session_all) and of the displayed items
        feature_seq = knowledge.features(browsed).T
        item_features = knowledge.features(item_list)
        
        # Integer attributes as in MATLAB, KB(list(f), :)' and KB(list, :), when KB exists
        if self.knowledge_base is not None:
            semantic_seq = self.knowledge_base[browsed - 1, :].T
            semantic_attributes = self.knowledge_base[item_list - 1, :]
        else:
            semantic_seq = semantic_attributes = None
        
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
            'semantic_attribute_seq': semantic_seq,   # KB(list(f), :)' (attributes x T) or None
            'semantic_attributes': semantic_attributes, # KB(list, :) or None
            'feature_seq': feature_seq,               # values of the browsed items (features x T, 0/1)
            'item_features': item_features,           # values of the displayed items (items x features, 0/1)
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
        if self.knowledge is None:
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
            'KB' (None with multi-valued attributes), 'knowledge' and 'n_usr';
            use main.reader_session_list for MATLAB's cell(:) order

        Raises:
            FileNotFoundError / ValueError: missing or invalid files (no fallback to fake data)
        """
        if self.knowledge is None:
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
            session['session_id'] = session_id  # order of the session for this user (1..3), as in gazedata.mat
            sessions[key] = session

        self.sessions = sessions
        self.semantic_attribute_seq = {k: s['semantic_attribute_seq'] for k, s in sessions.items()}
        self.semantic_attributes = {k: s['semantic_attributes'] for k, s in sessions.items()}
        return {
            'session': sessions,
            'semantic_attribute_seq': self.semantic_attribute_seq,
            'semantic_attributes': self.semantic_attributes,
            'KB': self.knowledge_base,
            'knowledge': self.knowledge,
            'n_usr': usr
        }

    def save_gazedata(self, data: Dict[str, Any], output_file: str = "gazedata.mat"):
        """
        Save sessions as a MATLAB gazedata.mat readable by main.load_gazedata_mat / main.load_gazedata_sessions

        The session structs (f, dur, list, task, usr, session_id, totalduration) are saved as an n_usr x n_task cell
        array indexed {usr, task}, as in ReadwithTimestamp.m. semantic_attribute_seq / semantic_attributes and KB
        are saved as well when every item has one value per attribute (KB exists).

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
        fields = ('f', 'dur', 'list', 'task', 'usr', 'session_id', 'totalduration')
        structs = np.empty((n_usr, n_task), dtype=object)
        for u in range(n_usr):
            for t in range(n_task):
                s = data['session'].get((u + 1, t + 1))
                structs[u, t] = np.zeros((0, 0)) if s is None else {
                    k: (np.asarray(s[k], dtype=float).reshape(-1, 1) if k in ('f', 'dur', 'list') else float(s[k]))
                    for k in fields if k in s}
        out = {'session': structs}
        if data.get('KB') is not None:
            for name in ('semantic_attribute_seq', 'semantic_attributes'):
                c = np.empty((n_usr, n_task), dtype=object)
                for u in range(n_usr):
                    for t in range(n_task):
                        c[u, t] = np.asarray(data[name].get((u + 1, t + 1), np.zeros((0, 0))), dtype=float)
                out[name] = c
            out['KB'] = np.asarray(data['KB'], dtype=float)
        scipy.io.savemat(output_file, out)
        print(f"Saved {len(keys)} sessions ({n_usr} users x {n_task} tasks) to {output_file}")


def main_read_gaze_data():
    """
    Read all sessions and save gazedata.mat (ReadwithTimestamp.m)

    Usage (from python-ver/):
      python src/data/gaze_reader.py --timestamp-list ../experiment/timestamp.list \
          --knowledge ../knowledge/knowledge.csv [--grid 3 4] --out gazedata.mat
      python main.py --mat gazedata.mat --knowledge ../knowledge/knowledge.csv --eval
    """
    import argparse
    parser = argparse.ArgumentParser(description="Read gaze data (ReadwithTimestamp.m)")
    parser.add_argument('--timestamp-list', required=True, help="timestamp.list (one session name per line)")
    parser.add_argument('--experiment-dir', default=None,
                        help="directory with <name>.csv / .list (default: directory of timestamp.list)")
    parser.add_argument('--knowledge', required=True, help="knowledge.csv (item attributes; see data.knowledge)")
    parser.add_argument('--grid', type=int, nargs=2, default=(3, 4), metavar=('ROWS', 'COLS'),
                        help="item grid on the screen (default: 3 4)")
    parser.add_argument('--out', default='gazedata.mat')
    args = parser.parse_args()

    reader = GazeDataReader(base_path='.', knowledge_csv_path=args.knowledge, grid=args.grid)
    reader.load_knowledge_base()
    data = reader.load_all_sessions(args.timestamp_list, args.experiment_dir)
    reader.save_gazedata(data, args.out)
    return data


if __name__ == "__main__":
    main_read_gaze_data()
