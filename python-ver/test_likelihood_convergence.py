#!/usr/bin/env python3
"""
Quick Test for pLSA Likelihood Convergence Plotting
This is a lightweight test for the likelihood convergence functionality.

NOTE: This is a standalone test for quick validation.
For comprehensive testing, use test_comprehensive_visualization.py

Purpose:
- Quick validation of likelihood convergence plotting
- Lightweight testing during development
- Minimal data (50 samples × 3 sessions) for fast execution
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.plsa import FourLayerPLSA
from visualization.plotter import AspectPlotter


def quick_likelihood_convergence_test():
    """Quick test for likelihood convergence plotting functionality"""
    print("🧪 Quick Test: pLSA Likelihood Convergence Plotting")
    print("=" * 50)
    print("📝 This is a lightweight test for development purposes")
    print("📋 For comprehensive testing, use test_comprehensive_visualization.py")
    print("")
    
    # Create directories
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize plotter
    plotter = AspectPlotter()
    
    # Create simple toy data: 3 sessions with 50 samples each
    print("📊 Creating toy data...")
    np.random.seed(3000)  # Different seed for convergence test
    
    sessions_data = []
    n_sessions = 3
    n_samples = 50
    n_features = 15  # 5 attributes × 3 values
    
    for session_id in range(n_sessions):
        # Create 15D one-hot encoded features
        mock_onehot = np.zeros((n_features, n_samples))
        
        for t in range(n_samples):
            for attr in range(5):
                value = np.random.randint(0, 3)  # 0, 1, or 2
                feature_idx = attr * 3 + value
                mock_onehot[feature_idx, t] = 1.0
        
        sessions_data.append(mock_onehot)
    
    print(f"   Created {len(sessions_data)} sessions with {n_samples} samples each")
    
    # Test different numbers of topics
    likelihood_histories = {}
    
    for n_topics in [2, 3, 4]:
        print(f"🔬 Training {n_topics}-topic pLSA model...")
        
        model = FourLayerPLSA(n_topics=n_topics)
        
        # Set identity observation model
        p_o_v = np.eye(n_features)
        model.set_observation_model(p_o_v)
        
        # Train model with more iterations to see convergence
        params, history = model.fit(sessions_data, train_lim=1e-5, max_itr=25)
        
        likelihood_histories[n_topics] = history
        
        print(f"   Iterations: {len(history)}")
        print(f"   Initial likelihood: {history[0]:.4f}")
        print(f"   Final likelihood: {history[-1]:.4f}")
        print(f"   Improvement: {history[-1] - history[0]:.4f}")
        
        # Plot individual convergence
        fig = plotter.plot_likelihood_convergence(
            likelihood_history=history,
            topic_num=n_topics,
            save_path=f"figures/likelihood_convergence_{n_topics}topics.png"
        )
        plt.close(fig)
        print(f"   ✅ Saved individual plot: likelihood_convergence_{n_topics}topics.png")
    
    # Plot comparison of all convergence curves
    print("📈 Creating convergence comparison plot...")
    fig = plotter.plot_multiple_likelihood_convergence(
        likelihood_histories=likelihood_histories,
        title="pLSA Convergence Comparison (Quick Test)",
        save_path="figures/likelihood_convergence_comparison.png"
    )
    plt.close(fig)
    print("   ✅ Saved comparison plot: likelihood_convergence_comparison.png")
    
    print("\n" + "=" * 50)
    print("🎯 Test Results:")
    print(f"   Tested {len(likelihood_histories)} different topic models")
    print(f"   Generated {len(likelihood_histories) + 1} plots")
    print("   📁 Check figures/ directory for generated plots")
    print("✅ Likelihood convergence plotting test completed!")
    
    return True


if __name__ == "__main__":
    print("🚀 Starting Quick Likelihood Convergence Test...")
    print("    ⚡ Lightweight test for development")
    print("    📋 Use test_comprehensive_visualization.py for full testing")
    
    success = quick_likelihood_convergence_test()
    
    if success:
        print("\n✅ Quick test passed!")
        print("🔗 Run test_comprehensive_visualization.py for complete testing")
    else:
        print("\n❌ Quick test failed!")
        
    exit(0 if success else 1)
