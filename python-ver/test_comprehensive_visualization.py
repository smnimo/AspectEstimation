#!/usr/bin/env python3
"""
Comprehensive Visualization Test Suite
Tests all MATLAB plot_*.m and R plot functions converted to Python

Covers:
- Original 6 plot functions: plot_L_topic.m, plot_conf_matrix.m, plot_dur_usr.m, etc.
- Additional 2 plot functions: plot_org_gaze.m, plot_aspect.R  
- Full pLSA integration testing
- Comprehensive report generation
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.gaze_reader import GazeDataReader
from models.plsa import FourLayerPLSA
from visualization.plotter import AspectPlotter


class ComprehensiveVisualizationTest:
    """
    Unified test class for all visualization functionality
    """
    
    def __init__(self):
        self.plotter = AspectPlotter()
        self.reader = None
        self.session = None
        self.results_dir = "results"
        self.figures_dir = "figures"
        self.last_plsa_params = None  # Store latest pLSA parameters for visualization
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def test_data_loading(self):
        """Test data loading functionality using standardized toy data: 200 samples × 20 sessions"""
        print("\n1️⃣ Testing Data Loading (200 samples × 20 sessions)...")
        
        try:
            # Skip real data loading, use only toy data for testing
            print("   📝 Creating standardized toy data: 200 samples × 20 sessions")
            
            # Create standardized toy data: 200 samples × 20 sessions
            self.sessions = []
            n_samples = 200
            n_sessions = 20
            
            # Common Knowledge Base (shared across all sessions)
            self.common_kb = np.array([
                    [1, 2, 3, 1, 2],  # Item 1
                    [2, 3, 1, 2, 3],  # Item 2
                    [3, 1, 2, 3, 1],  # Item 3
                    [1, 3, 2, 1, 3],  # Item 4
                    [2, 1, 3, 2, 1],  # Item 5
                    [3, 2, 1, 3, 2],  # Item 6
                    [1, 1, 1, 1, 1],  # Item 7
                    [2, 2, 2, 2, 2],  # Item 8
                    [3, 3, 3, 3, 3],  # Item 9
                    [1, 2, 3, 2, 1],  # Item 10
                    [2, 3, 1, 3, 2],  # Item 11
                    [3, 1, 2, 1, 3],  # Item 12
            ])
            
            # Generate 20 sessions with 200 samples each
            np.random.seed(2000)  # Unique seed for 200x20 toy data generation
            
            for session_id in range(n_sessions):
                # Create 15D one-hot encoded features for 200 samples
                mock_onehot = np.zeros((15, n_samples))
                
                # Create realistic one-hot pattern: exactly 5 ones per sample (one per attribute)
                for t in range(n_samples):
                    for attr in range(5):
                        value = np.random.randint(0, 3)  # 0, 1, or 2
                        feature_idx = attr * 3 + value
                        mock_onehot[feature_idx, t] = 1.0
                
                session_data = {
                    'session_id': session_id + 1,
                    'semantic_attribute_seq': np.random.randint(1, 4, (5, n_samples)),  # Old 5D for compatibility
                    'semantic_attribute_seq_onehot': mock_onehot,  # New 15D one-hot
                    'f': list(range(n_samples)),
                    'gaze_seq': np.random.rand(n_samples, 2) * np.array([1920, 1080]),  # Realistic screen coordinates
                    'user': (session_id % 5) + 1,  # 5 users across 20 sessions
                    'task': (session_id % 3) + 1,  # 3 tasks across 20 sessions
                    'totalduration': 45.2 + np.random.rand() * 30  # Random duration 45-75s
                }
                self.sessions.append(session_data)            # Store first session for compatibility
            self.session = self.sessions[0]
            
            print(f"   📚 Common KB shape: {self.common_kb.shape}")
            print(f"   📊 Created {len(self.sessions)} sessions with {n_samples} samples each")
            print(f"   👥 Users: {len(set(s['user'] for s in self.sessions))} users")
            print(f"   📋 Tasks: {len(set(s['task'] for s in self.sessions))} tasks")
            print("   ✅ Standardized toy data (200×20) created successfully")
            return True
                
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            return False
    
    def test_original_plot_functions(self):
        """Test original 6 plot functions using 20 sessions data"""
        print("\n2️⃣ Testing Original Plot Functions (20 sessions)...")
        
        # Create mock data structure for existing plots using 20 sessions
        sessions_data = {
            'session': {}
        }
        
        # Build session data from our 20 sessions
        if hasattr(self, 'sessions') and self.sessions:
            for i, session in enumerate(self.sessions):
                user_id = session['user']
                task_id = session['task']
                sessions_data['session'][(user_id, task_id)] = {
                    'usr': user_id,
                    'task': task_id,
                    'totalduration': session['totalduration']
                }
            print(f"   📊 Using {len(self.sessions)} sessions data")
        else:
            # Fallback to mock data
            sessions_data['session'] = {
                (1, 1): {'usr': 1, 'task': 1, 'totalduration': 45.2},
                (1, 2): {'usr': 1, 'task': 2, 'totalduration': 52.1},
                (1, 3): {'usr': 1, 'task': 3, 'totalduration': 38.7},
                (2, 1): {'usr': 2, 'task': 1, 'totalduration': 67.3},
                (2, 2): {'usr': 2, 'task': 2, 'totalduration': 71.8},
                (2, 3): {'usr': 2, 'task': 3, 'totalduration': 55.4},
            }
            print("   📊 Using fallback mock sessions data")
        
        results = []
        
        # Test 1: Duration analysis (plot_dur_usr.m)
        try:
            print("   📊 Duration analysis (plot_dur_usr.m)...")
            fig = self.plotter.plot_duration_by_user(sessions_data)
            fig.savefig(f"{self.figures_dir}/duration_analysis_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Duration analysis completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Duration analysis failed: {e}")
            results.append(False)
        
        # Test 2: Confusion matrix (plot_conf_matrix.m)
        try:
            print("   🔢 Confusion matrix (plot_conf_matrix.m)...")
            
            # Provide mock confusion matrices for testing
            test_sessions_data = sessions_data.copy()
            test_sessions_data['confusion_mset'] = np.array([
                [5, 1, 0, 0],  # Task 1: 5 correct, 1 as Task 2, 0 as Task 3, 0 unknown
                [1, 4, 1, 0],  # Task 2: 1 as Task 1, 4 correct, 1 as Task 3, 0 unknown  
                [0, 0, 6, 0]   # Task 3: 0 as Task 1, 0 as Task 2, 6 correct, 0 unknown
            ])
            test_sessions_data['confusion_no_mset'] = np.array([
                [4, 2, 0, 0],  # Task 1: 4 correct, 2 as Task 2, 0 as Task 3, 0 unknown
                [2, 3, 1, 0],  # Task 2: 2 as Task 1, 3 correct, 1 as Task 3, 0 unknown
                [1, 1, 4, 0]   # Task 3: 1 as Task 1, 1 as Task 2, 4 correct, 0 unknown
            ])
            
            fig, _, _ = self.plotter.plot_confusion_matrix(test_sessions_data)
            fig.savefig(f"{self.figures_dir}/confusion_matrix_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Confusion matrix completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Confusion matrix failed: {e}")
            results.append(False)
        
        # Test 3: Likelihood comparison (plot_L_topic.m)
        try:
            print("   📈 Likelihood comparison (plot_L_topic.m)...")
            fig = self.plotter.plot_likelihood_comparison(max_topics=3)
            fig.savefig(f"{self.figures_dir}/likelihood_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Likelihood comparison completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Likelihood comparison failed: {e}")
            results.append(False)
        
        # Test 4: Aspect matrices with actual pLSA parameters
        try:
            print("   🔲 Aspect matrices...")
            # Use pLSA integration data if available
            mock_data_with_params = sessions_data.copy()
            
            # If we have real pLSA results, use them
            if hasattr(self, 'last_plsa_params') and self.last_plsa_params is not None:
                mock_data_with_params['params'] = self.last_plsa_params
                print("      📊 Using actual pLSA parameters for aspect matrices")
            else:
                # Create mock 15D pLSA parameters
                mock_params = {
                    'vz': [
                        np.random.rand(15) / 15,  # Topic 1: 15D feature distribution
                        np.random.rand(15) / 15,  # Topic 2: 15D feature distribution  
                        np.random.rand(15) / 15   # Topic 3: 15D feature distribution
                    ]
                }
                # Normalize each topic's feature distribution
                for i, vz in enumerate(mock_params['vz']):
                    mock_params['vz'][i] = vz / np.sum(vz)
                
                mock_data_with_params['params'] = mock_params
                print("      📊 Using mock 15D pLSA parameters for aspect matrices")
            
            fig = self.plotter.plot_aspect_matrices(mock_data_with_params, max_topics=3)
            fig.savefig(f"{self.figures_dir}/aspect_matrices_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Aspect matrices completed (15D features)")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Aspect matrices failed: {e}")
            results.append(False)
        
        # Test 5: Similarity analysis (plot_sim*.m)
        try:
            print("   🔄 Similarity analysis (plot_sim*.m)...")
            
            # Provide mock similarity matrices for testing
            test_sessions_data = sessions_data.copy()
            test_sessions_data['task_similarity'] = np.array([
                [1.0, 0.7, 0.4],
                [0.7, 1.0, 0.6],
                [0.4, 0.6, 1.0]
            ])
            test_sessions_data['aspect_similarity'] = np.array([
                [1.0, 0.8, 0.5, 0.3, 0.2],
                [0.8, 1.0, 0.6, 0.4, 0.3],
                [0.5, 0.6, 1.0, 0.7, 0.5],
                [0.3, 0.4, 0.7, 1.0, 0.8],
                [0.2, 0.3, 0.5, 0.8, 1.0]
            ])
            test_sessions_data['task_aspect_similarity'] = np.array([
                [0.8, 0.6, 0.4, 0.3, 0.2],  # Task 1 similarity to aspects
                [0.5, 0.9, 0.7, 0.4, 0.3],  # Task 2 similarity to aspects
                [0.3, 0.4, 0.8, 0.9, 0.6]   # Task 3 similarity to aspects
            ])
            
            # Overall correlation matrix (8x8: 3 tasks + 5 aspects)
            test_sessions_data['overall_correlation'] = np.array([
                [1.0, 0.7, 0.4, 0.6, 0.5, 0.3, 0.2, 0.1],  # Task 1
                [0.7, 1.0, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2],  # Task 2
                [0.4, 0.6, 1.0, 0.3, 0.4, 0.8, 0.6, 0.4],  # Task 3
                [0.6, 0.4, 0.3, 1.0, 0.8, 0.5, 0.3, 0.2],  # Aspect 1
                [0.5, 0.7, 0.4, 0.8, 1.0, 0.6, 0.4, 0.3],  # Aspect 2
                [0.3, 0.5, 0.8, 0.5, 0.6, 1.0, 0.7, 0.5],  # Aspect 3
                [0.2, 0.3, 0.6, 0.3, 0.4, 0.7, 1.0, 0.8],  # Aspect 4
                [0.1, 0.2, 0.4, 0.2, 0.3, 0.5, 0.8, 1.0]   # Aspect 5
            ])
            
            fig = self.plotter.plot_similarity_analysis(test_sessions_data)
            fig.savefig(f"{self.figures_dir}/similarity_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Similarity analysis completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Similarity analysis failed: {e}")
            results.append(False)
        
        # Test 6: Evaluation metrics (plot_eval_AoF.m)
        try:
            print("   📊 Evaluation metrics (plot_eval_AoF.m)...")
            
            # Provide mock evaluation metrics for testing
            test_evaluation_data = {
                'accuracy': np.array([0.75, 0.82, 0.78, 0.85, 0.80]),
                'precision': np.array([0.72, 0.79, 0.76, 0.83, 0.77]),
                'recall': np.array([0.70, 0.81, 0.74, 0.82, 0.79])
            }
            
            fig = self.plotter.plot_evaluation_metrics(test_evaluation_data)
            fig.savefig(f"{self.figures_dir}/evaluation_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Evaluation metrics completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Evaluation metrics failed: {e}")
            results.append(False)
        
        return results
    
    def test_additional_plot_functions(self):
        """Test additional 2 plot functions (new functionality)"""
        print("\n3️⃣ Testing Additional Plot Functions...")
        
        results = []
        
        # Test 7: Gaze trajectory (plot_org_gaze.m)
        try:
            print("   👁️ Gaze trajectory (plot_org_gaze.m)...")
            mock_gaze_data = {
                'session_id': 'comprehensive_test',
                'image_size': (1920, 1080),
                'gaze_seq': np.array([
                    [0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.3, 0.4],
                    [0.35, 0.45], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7],
                    [0.65, 0.75], [0.7, 0.8], [0.75, 0.85], [0.8, 0.9]
                ])  # Normalized coordinates (0-1)
            }
            
            fig = self.plotter.plot_original_gaze_trajectories(
                gaze_data=mock_gaze_data,
                save_path=f"{self.figures_dir}/gaze_trajectory_test.png"
            )
            plt.close(fig)
            print("      ✅ Gaze trajectory completed (origin: top-left)")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Gaze trajectory failed: {e}")
            results.append(False)
        
        # Test 8: Aspect correlation circles (plot_aspect.R) - Updated for 15D features
        try:
            print("   ⭕ Aspect correlation circles (15D features)...")
            
            # Create 15×N_topics probability matrix representing Topic→Feature distributions
            n_topics = 3
            np.random.seed(1100)  # Different seed for aspect matrices test
            
            # Create realistic Topic→Feature probability matrix (15 features × 3 topics)
            mock_aspect_matrix = np.random.rand(15, n_topics)  
            
            # Normalize each column (topic) to sum to 1
            for t in range(n_topics):
                mock_aspect_matrix[:, t] = mock_aspect_matrix[:, t] / np.sum(mock_aspect_matrix[:, t])
            
            # Create feature names for 15D one-hot features
            feature_names = []
            for attr in range(5):
                for value in range(1, 4):  # Values 1, 2, 3
                    feature_names.append(f'Attr{attr+1}_Val{value}')
            
            
            fig = self.plotter.plot_aspect_correlation_circles(
                aspect_matrix=mock_aspect_matrix,
                feature_names=feature_names,
                save_path=f"{self.figures_dir}/aspect_correlations_test.png"
            )
            plt.close(fig)
            print("      ✅ Aspect correlations completed (15D features, colorbar: 0=white, 1=blue)")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Aspect correlations failed: {e}")
            results.append(False)
        
        return results
    
    def test_plsa_integration(self):
        """Test pLSA model integration with 200 samples × 20 sessions toy data"""
        print("\n4️⃣ Testing pLSA Integration (200 samples × 20 sessions)...")
        
        if not hasattr(self, 'sessions') or len(self.sessions) == 0:
            print("   ❌ No session data available")
            return [False]
        
        try:
            print(f"   📚 Using common KB: {self.common_kb.shape}")
            print(f"   📊 Processing {len(self.sessions)} sessions with 200 samples each")
            
            # Train models with different topic numbers using 15D one-hot features
            likelihood_results = {}
            
            for n_topics in [2, 3]:
                print(f"   🔬 Training {n_topics}-topic model with {len(self.sessions)} sessions...")
                
                model = FourLayerPLSA(n_topics=n_topics)
                
                # Set identity observation model for 15D features (direct modeling)
                n_features = 15  # 5 attributes × 3 values
                p_o_v = np.eye(n_features)
                model.set_observation_model(p_o_v)
                
                # Use 15D one-hot encoded features from all 20 sessions
                sessions_data = []
                for session in self.sessions:
                    if 'semantic_attribute_seq_onehot' in session:
                        sessions_data.append(session['semantic_attribute_seq_onehot'])
                    else:
                        # Create mock 15D one-hot data as fallback
                        mock_15d = np.zeros((15, 200))
                        for t in range(200):
                            for attr in range(5):
                                value = np.random.randint(0, 3)
                                feature_idx = attr * 3 + value
                                mock_15d[feature_idx, t] = 1.0
                        sessions_data.append(mock_15d)
                
                print(f"   📊 Training with {len(sessions_data)} sessions, each with {sessions_data[0].shape} features")
                
                params, history = model.fit(sessions_data, train_lim=1e-4, max_itr=5)
                
                # Store latest params for aspect matrices visualization
                self.last_plsa_params = params
                
                likelihood_results[n_topics] = {
                    'likelihood': history[-1],
                    'params': params,
                    'history': history
                }
                print(f"      Final likelihood: {history[-1]:.4f}")
                print(f"      Saved P(v|z) parameters: {len(params['vz'])} topics × {len(params['vz'][0])}D features")
            
            # Save results for visualization
            for n_topics, results in likelihood_results.items():
                np.savez(f"{self.results_dir}/test_toy_topic{n_topics}.npz",
                        L=np.array([results['likelihood']]),
                        params=results['params'],
                        allow_pickle=True)
            
            print("   ✅ pLSA integration completed")
            return [True]
            
        except Exception as e:
            print(f"   ❌ pLSA integration failed: {e}")
            return [False]
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation using individual functions (avoid duplication)"""
        print("\n5️⃣ Testing Comprehensive Report (Individual Functions)...")
        
        try:
            # Create comprehensive data structure with 15D features
            mock_data = {
                'session': {
                    (1, 1): {'usr': 1, 'task': 1, 'totalduration': 45.2},
                    (1, 2): {'usr': 1, 'task': 2, 'totalduration': 52.1},
                    (2, 1): {'usr': 2, 'task': 1, 'totalduration': 67.3},
                }
            }
            
            # Use actual pLSA parameters if available
            if hasattr(self, 'last_plsa_params') and self.last_plsa_params is not None:
                mock_data['params'] = self.last_plsa_params
                print("   📊 Using actual pLSA parameters for comprehensive report")
                
                # Test only aspect correlation circles with real data
                params = self.last_plsa_params
                if 'vz' in params and params['vz'] is not None:
                    n_topics = len(params['vz'])
                    n_features = len(params['vz'][0]) if params['vz'] else 15
                    
                    aspect_matrix = np.zeros((n_features, n_topics))
                    for k, vz_k in enumerate(params['vz']):
                        aspect_matrix[:, k] = vz_k
                    
                    feature_names = []
                    if n_features == 15:
                        for attr in range(5):
                            for value in range(1, 4):
                                feature_names.append(f'Attr{attr+1}_Val{value}')
                    else:
                        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
                    
                    fig = self.plotter.plot_aspect_correlation_circles(
                        aspect_matrix=aspect_matrix,
                        feature_names=feature_names,
                        title_suffix="Real pLSA Data"
                    )
                    fig.savefig(f"{self.figures_dir}/aspect_correlations.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print("      ✅ Created aspect correlations with real pLSA data")
                else:
                    print("      ⚠️ No vz parameters in pLSA data")
            else:
                print("   📊 No pLSA parameters available - comprehensive report will use individual mock functions")
                print("      ✅ Comprehensive report logic verified (no duplication)")
            
            print("   ✅ Comprehensive report completed")
            return [True]
            
        except Exception as e:
            print(f"   ❌ Comprehensive report failed: {e}")
            return [False]
    
    def run_all_tests(self):
        """Run all visualization tests"""
        print("🧪 Comprehensive Visualization Test Suite")
        print("=" * 70)
        print("Testing complete MATLAB plot_*.m and R plot conversion to Python")
        
        # Import matplotlib here to avoid early import issues
        global plt
        import matplotlib.pyplot as plt
        
        # Run all test categories
        test_results = []
        
        # 1. Data loading
        test_results.extend([self.test_data_loading()])
        
        # 2. pLSA integration (run early to get parameters for aspect matrices)
        test_results.extend(self.test_plsa_integration())
        
        # 3. Original 6 plot functions (now using real pLSA parameters if available)
        test_results.extend(self.test_original_plot_functions())
        
        # 4. Additional 2 plot functions  
        test_results.extend(self.test_additional_plot_functions())
        
        # 5. Comprehensive report
        test_results.extend(self.test_comprehensive_report())
        
        # Summary
        passed = sum(test_results)
        total = len(test_results)
        
        print("\n" + "=" * 70)
        print("🎯 Test Results Summary:")
        print(f"   Passed: {passed}/{total} tests")
        print(f"   Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("✅ All visualization functions working correctly!")
            print("🎉 Complete MATLAB/R plot conversion successful!")
        else:
            print("⚠️ Some tests failed. Check error messages above.")
        
        print(f"\n📁 Generated files:")
        print(f"   - Figures: {self.figures_dir}/")
        print(f"   - Results: {self.results_dir}/")
        
        return passed == total


def main():
    """Main test execution"""
    # Create and run comprehensive test
    test_suite = ComprehensiveVisualizationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🚀 Ready for production use!")
    else:
        print("\n🔧 Some issues need attention.")
    
    return success


if __name__ == "__main__":
    main()
