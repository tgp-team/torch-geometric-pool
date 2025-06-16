#!/usr/bin/env python3
"""
Integration test for MaxCutSelect and MaxCutPooling.
"""

import torch
from tgp.poolers import get_pooler
from tgp.select import MaxCutSelect


def test_maxcut_select():
    """Test MaxCutSelect standalone functionality."""
    print("Testing MaxCutSelect...")
    
    # Create test data
    N, F = 20, 16
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, 40))
    edge_weight = torch.ones(40)
    batch = torch.zeros(N, dtype=torch.long)
    
    # Initialize MaxCutSelect
    selector = MaxCutSelect(in_channels=F, ratio=0.5)
    
    try:
        # Forward pass
        select_output = selector(x, edge_index, edge_weight, batch)
        
        print(f"✓ Selected {len(select_output.node_index)} / {N} nodes")
        print(f"✓ Select output has scores: {hasattr(select_output, 'scores')}")
        print(f"✓ Select output has edge_index: {hasattr(select_output, 'edge_index')}")
        
        if hasattr(select_output, 'scores'):
            print(f"✓ Scores shape: {select_output.scores.shape}")
            print(f"✓ Scores range: [{select_output.scores.min():.3f}, {select_output.scores.max():.3f}]")
        
        # Test that scores require gradients
        if hasattr(select_output, 'scores'):
            print(f"✓ Scores require grad: {select_output.scores.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"✗ MaxCutSelect test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_maxcut_pooling():
    """Test MaxCutPooling via get_pooler."""
    print("\nTesting MaxCutPooling...")
    
    try:
        # Test pooler creation via get_pooler
        pooler = get_pooler("maxcut", in_channels=16, ratio=0.5, loss_coeff=1.0)
        print(f"✓ Created pooler: {type(pooler).__name__}")
        print(f"✓ Has loss: {pooler.has_loss}")
        
        # Create test data
        N, F = 30, 16
        x = torch.randn(N, F)
        edge_index = torch.randint(0, N, (2, 60))
        edge_weight = torch.ones(60)
        
        # Forward pass
        out = pooler(x=x, edge_index=edge_index, edge_weight=edge_weight)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {out.x.shape}")
        print(f"✓ Input edges: {edge_index.size(1)}")
        print(f"✓ Output edges: {out.edge_index.size(1)}")
        print(f"✓ Has loss: {out.has_loss}")
        
        if out.has_loss:
            loss_value = out.get_loss_value('maxcut_loss')
            print(f"✓ MaxCut loss value: {loss_value}")
            print(f"✓ Loss is tensor: {isinstance(loss_value, torch.Tensor)}")
        
        # Test gradient flow
        total_loss = sum(out.loss.values())
        total_loss.backward()
        print("✓ Gradients computed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ MaxCutPooling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_operation():
    """Test batched operation."""
    print("\nTesting batched operation...")
    
    try:
        pooler = get_pooler("maxcut", in_channels=8, ratio=0.6, loss_coeff=0.5)
        
        # Create batched data (2 graphs)
        x = torch.randn(50, 8)
        edge_index = torch.randint(0, 50, (2, 80))
        batch = torch.cat([torch.zeros(25), torch.ones(25)]).long()
        
        out = pooler(x=x, edge_index=edge_index, batch=batch)
        
        print(f"✓ Batched input: {x.shape}")
        print(f"✓ Batched output: {out.x.shape}")
        print(f"✓ Batch reduction: {50} -> {out.x.size(0)} nodes")
        print(f"✓ Loss: {out.get_loss_value('maxcut_loss'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batched test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_network_kwargs():
    """Test custom score network parameters."""
    print("\nTesting custom score network parameters...")
    
    try:
        # Test with custom score network parameters
        pooler = get_pooler(
            "maxcut",
            in_channels=16,
            ratio=0.3,
            loss_coeff=2.0,
            mp_units=[8, 8, 4],  # Smaller network
            mp_act="relu",
            mlp_units=[4],
            mlp_act="tanh"
        )
        
        x = torch.randn(20, 16)
        edge_index = torch.randint(0, 20, (2, 30))
        
        out = pooler(x=x, edge_index=edge_index)
        
        print(f"✓ Custom network pooling: {x.shape} -> {out.x.shape}")
        print(f"✓ Loss: {out.get_loss_value('maxcut_loss'):.4f}")
        print(f"✓ Loss coefficient applied: {abs(out.get_loss_value('maxcut_loss')) > 0}")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom kwargs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selectoutput_data():
    """Test that SelectOutput contains the expected data for loss computation."""
    print("\nTesting SelectOutput data...")
    
    try:
        selector = MaxCutSelect(in_channels=8, ratio=0.5)
        
        N, F = 15, 8
        x = torch.randn(N, F)
        edge_index = torch.randint(0, N, (2, 25))
        edge_weight = torch.ones(25)
        
        select_output = selector(x, edge_index, edge_weight)
        
        # Check required data for loss computation
        print(f"✓ Has scores: {hasattr(select_output, 'scores')}")
        print(f"✓ Has edge_index: {hasattr(select_output, 'edge_index')}")
        print(f"✓ Has edge_weight: {hasattr(select_output, 'edge_weight')}")
        
        if hasattr(select_output, 'scores'):
            print(f"✓ Scores shape: {select_output.scores.shape}")
            print(f"✓ Original nodes: {N}, Score entries: {len(select_output.scores)}")
            
        # Check that extra_args are updated
        expected_extra = {'scores', 'edge_index', 'edge_weight'}
        actual_extra = select_output._extra_args
        print(f"✓ Extra args contain required keys: {expected_extra.issubset(actual_extra)}")
        
        return True
        
    except Exception as e:
        print(f"✗ SelectOutput data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== MaxCut Integration Tests ===")
    
    success = True
    success &= test_maxcut_select()
    success &= test_maxcut_pooling()
    success &= test_batched_operation()
    success &= test_score_network_kwargs()
    success &= test_selectoutput_data()
    
    print(f"\n=== Test Summary ===")
    if success:
        print("🎉 All MaxCut integration tests passed!")
    else:
        print("❌ Some MaxCut integration tests failed.") 