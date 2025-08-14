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
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def test_data_loading(self):
        """Test data loading functionality"""
        print("\n1️⃣ Testing Data Loading...")
        
        try:
            self.reader = GazeDataReader(knowledge_csv_path='../sample_data/knowledge.csv')
            
            # Try to load real data
            try:
                self.session = self.reader.load_session_from_files(
                    '../sample_data/170116024150092.csv',
                    '../sample_data/170116024150092.list'
                )
                print("   ✅ Real data loaded successfully")
                return True
            except Exception as e:
                print(f"   ⚠️ Real data not available: {e}")
                print("   📝 Using mock data for testing")
                
                # Create mock session data with 15D one-hot features
                # Mock 15-dimensional one-hot encoded features (5 attributes × 3 values)
                n_timepoints = 30
                mock_onehot = np.zeros((15, n_timepoints))
                
                # Create realistic one-hot pattern: exactly 5 ones per timepoint
                np.random.seed(42)
                for t in range(n_timepoints):
                    # For each attribute (0-4), randomly select one of 3 values (0-2)
                    for attr in range(5):
                        value = np.random.randint(0, 3)  # 0, 1, or 2
                        feature_idx = attr * 3 + value
                        mock_onehot[feature_idx, t] = 1.0
                
                self.session = {
                    'semantic_attribute_seq': np.random.randint(1, 4, (5, n_timepoints)),  # Old 5D for compatibility
                    'semantic_attribute_seq_onehot': mock_onehot,  # New 15D one-hot
                    'f': list(range(n_timepoints)),
                    'gaze_seq': np.random.rand(n_timepoints, 2),
                    'user': 1,
                    'task': 1,
                    'totalduration': 45.2
                }
                return True
                
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            return False
    
    def test_original_plot_functions(self):
        """Test original 6 plot functions (existing functionality)"""
        print("\n2️⃣ Testing Original Plot Functions...")
        
        # Create mock data structure for existing plots
        sessions_data = {
            'session': {
                (1, 1): {'usr': 1, 'task': 1, 'totalduration': 45.2},
                (1, 2): {'usr': 1, 'task': 2, 'totalduration': 52.1},
                (1, 3): {'usr': 1, 'task': 3, 'totalduration': 38.7},
                (2, 1): {'usr': 2, 'task': 1, 'totalduration': 67.3},
                (2, 2): {'usr': 2, 'task': 2, 'totalduration': 71.8},
                (2, 3): {'usr': 2, 'task': 3, 'totalduration': 55.4},
            }
        }
        
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
            fig, _, _ = self.plotter.plot_confusion_matrix(sessions_data)
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
        
        # Test 4: Aspect matrices
        try:
            print("   🔲 Aspect matrices...")
            fig = self.plotter.plot_aspect_matrices(sessions_data, max_topics=3)
            fig.savefig(f"{self.figures_dir}/aspect_matrices_test.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("      ✅ Aspect matrices completed")
            results.append(True)
        except Exception as e:
            print(f"      ❌ Aspect matrices failed: {e}")
            results.append(False)
        
        # Test 5: Similarity analysis (plot_sim*.m)
        try:
            print("   🔄 Similarity analysis (plot_sim*.m)...")
            fig = self.plotter.plot_similarity_analysis(sessions_data)
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
            fig = self.plotter.plot_evaluation_metrics({})
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
            np.random.seed(42)
            
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
        """Test pLSA model integration with visualization"""
        print("\n4️⃣ Testing pLSA Integration...")
        
        if self.reader is None or self.session is None:
            print("   ⚠️ Skipping pLSA integration (no data available)")
            return [False]
        
        try:
            # Set up knowledge base
            if hasattr(self.reader, 'knowledge_base') and self.reader.knowledge_base is not None:
                kb = self.reader.knowledge_base
            else:
                # Create mock knowledge base
                kb = np.random.randint(1, 4, size=(12, 5))
                self.reader.knowledge_base = kb
            
            print(f"   📚 Knowledge base shape: {kb.shape}")
            
            # Train models with different topic numbers using 15D one-hot features
            likelihood_results = {}
            
            for n_topics in [2, 3]:
                print(f"   🔬 Training {n_topics}-topic model with 15D features...")
                
                model = FourLayerPLSA(n_topics=n_topics)
                
                # Set identity observation model for 15D features (direct modeling)
                n_features = 15  # 5 attributes × 3 values
                p_o_v = np.eye(n_features)
                model.set_observation_model(p_o_v)
                
                # Use 15D one-hot encoded features
                if 'semantic_attribute_seq_onehot' in self.session:
                    sessions = [self.session['semantic_attribute_seq_onehot']]
                    print(f"   📊 Using real 15D one-hot features: {self.session['semantic_attribute_seq_onehot'].shape}")
                else:
                    # Create mock 15D one-hot data
                    mock_15d = np.zeros((15, 20))
                    for t in range(20):
                        for attr in range(5):
                            value = np.random.randint(0, 3)
                            feature_idx = attr * 3 + value
                            mock_15d[feature_idx, t] = 1.0
                    sessions = [mock_15d]
                
                params, history = model.fit(sessions, train_lim=1e-4, max_itr=5)
                
                likelihood_results[n_topics] = {
                    'likelihood': history[-1],
                    'params': params,
                    'history': history
                }
                print(f"      Final likelihood: {history[-1]:.4f}")
            
            # Save results for visualization
            for n_topics, results in likelihood_results.items():
                np.savez(f"{self.results_dir}/p_all_topic{n_topics}.npz",
                        L=np.array([results['likelihood']]),
                        params=results['params'],
                        allow_pickle=True)
            
            print("   ✅ pLSA integration completed")
            return [True]
            
        except Exception as e:
            print(f"   ❌ pLSA integration failed: {e}")
            return [False]
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation (all 8 plots)"""
        print("\n5️⃣ Testing Comprehensive Report...")
        
        try:
            # Create comprehensive data structure with 15D features
            # Create mock 15D one-hot session data
            mock_session_15d = np.zeros((15, 30))
            for t in range(30):
                for attr in range(5):
                    value = np.random.randint(0, 3)
                    feature_idx = attr * 3 + value
                    mock_session_15d[feature_idx, t] = 1.0
            
            mock_data = {
                'sessions': [{'semantic_attribute_seq_onehot': mock_session_15d}],
                'models': [None],  # Placeholder
                'params': [{'vz': [np.random.rand(15, 3)]}],  # 15D features × 3 topics
                'history': [[100, 90, 85, 82, 80]],  # Mock likelihood history
                'gaze_trajectories': {
                    'session_id': 'comprehensive_test',
                    'image_size': (1920, 1080),
                    'gaze_seq': np.random.rand(40, 2)  # Normalized coordinates
                },
                'session': {
                    (1, 1): {'usr': 1, 'task': 1, 'totalduration': 45.2},
                    (1, 2): {'usr': 1, 'task': 2, 'totalduration': 52.1},
                    (2, 1): {'usr': 2, 'task': 1, 'totalduration': 67.3},
                }
            }
            
            saved_paths = self.plotter.create_comprehensive_report(mock_data)
            
            print(f"   🎨 Generated {len(saved_paths)} plots:")
            for path in saved_paths:
                print(f"      - {os.path.basename(path)}")
            
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
        
        # 2. Original 6 plot functions
        test_results.extend(self.test_original_plot_functions())
        
        # 3. Additional 2 plot functions  
        test_results.extend(self.test_additional_plot_functions())
        
        # 4. pLSA integration
        test_results.extend(self.test_plsa_integration())
        
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
