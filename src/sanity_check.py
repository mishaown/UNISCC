"""
UniSCC Model Sanity Check

Comprehensive validation script to test model correctness before training.
Can be imported in notebook for interactive testing.

Usage:
    # In notebook or script
    from sanity_check import run_all_checks, test_component
    
    # Run all checks
    results = run_all_checks()
    
    # Test specific component
    test_component('encoder')
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys

# Import model components
try:
    from uniscc import UniSCC, UniSCCConfig, build_uniscc
    from encoder import UniSCCEncoder
    from tdt import TemporalDifferenceTransformer
    from change_head import UnifiedChangeHead
    from caption_decoder import ChangeGuidedCaptionDecoder
    from lsp import LearnableSemanticPrompts
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Make sure all model files are in the same directory")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test(name: str, status: str, message: str = ""):
    """Print colored test result"""
    if status == "PASS":
        color = Colors.OKGREEN
        symbol = "✓"
    elif status == "FAIL":
        color = Colors.FAIL
        symbol = "✗"
    elif status == "WARN":
        color = Colors.WARNING
        symbol = "!"
    else:
        color = Colors.OKBLUE
        symbol = "→"
    
    print(f"{color}{symbol} {name}{Colors.ENDC}", end="")
    if message:
        print(f" - {message}")
    else:
        print()


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


# =============================================================================
# Component Tests
# =============================================================================

def test_encoder():
    """Test encoder module"""
    print_section("Testing Encoder")
    
    try:
        # Create encoder
        encoder = UniSCCEncoder(
            backbone="swin_base_patch4_window7_224",
            pretrained=False,  # Don't download for testing
            feature_dim=512,
            img_size=256,
            use_temporal_embed=True
        )
        print_test("Encoder initialization", "PASS")
        
        # Test forward pass
        B, H, W = 2, 256, 256
        img_t0 = torch.randn(B, 3, H, W)
        img_t1 = torch.randn(B, 3, H, W)
        
        outputs = encoder(img_t0, img_t1)
        print_test("Encoder forward pass", "PASS")
        
        # Check outputs
        assert 'features_t0' in outputs, "Missing features_t0"
        assert 'features_t1' in outputs, "Missing features_t1"
        print_test("Output keys present", "PASS")
        
        feat_t0 = outputs['features_t0']
        feat_t1 = outputs['features_t1']
        
        # Check shapes
        assert feat_t0.dim() == 4, f"Expected 4D tensor, got {feat_t0.dim()}D"
        assert feat_t0.shape[0] == B, f"Batch size mismatch: {feat_t0.shape[0]} vs {B}"
        assert feat_t0.shape[1] == 512, f"Feature dim mismatch: {feat_t0.shape[1]} vs 512"
        print_test("Feature shapes", "PASS", f"feat_t0: {tuple(feat_t0.shape)}, feat_t1: {tuple(feat_t1.shape)}")
        
        # Check temporal difference
        assert not torch.allclose(feat_t0, feat_t1), "Features identical (temporal embedding not working?)"
        print_test("Temporal embeddings", "PASS", "Features differ between t0 and t1")
        
        return True
    except Exception as e:
        print_test("Encoder test", "FAIL", str(e))
        return False


def test_tdt():
    """Test Temporal Difference Transformer"""
    print_section("Testing TDT")
    
    try:
        # Create TDT
        tdt = TemporalDifferenceTransformer(
            dim=512,
            num_heads=8,
            num_layers=3,
            dropout=0.1
        )
        print_test("TDT initialization", "PASS")
        
        # Test forward pass
        B, C, H, W = 2, 512, 32, 32
        feat_t0 = torch.randn(B, C, H, W)
        feat_t1 = torch.randn(B, C, H, W)
        
        diff_features = tdt(feat_t0, feat_t1)
        print_test("TDT forward pass", "PASS")
        
        # Check shape
        assert diff_features.shape == (B, C, H, W), f"Shape mismatch: {diff_features.shape}"
        print_test("Output shape", "PASS", f"diff_features: {tuple(diff_features.shape)}")
        
        # Check that it's not just subtraction
        simple_diff = feat_t0 - feat_t1
        assert not torch.allclose(diff_features, simple_diff, atol=1e-2), "TDT output too close to simple subtraction"
        print_test("Cross-temporal attention", "PASS", "Output differs from simple subtraction")
        
        return True
    except Exception as e:
        print_test("TDT test", "FAIL", str(e))
        return False


def test_change_head():
    """Test Unified Change Head"""
    print_section("Testing Change Head")
    
    try:
        # Create change head
        change_head = UnifiedChangeHead(
            in_channels=512,
            hidden_channels=256,
            scd_classes=7,
            bcd_classes=3
        )
        print_test("Change Head initialization", "PASS")
        
        # Test SCD mode (SECOND-CC)
        B, C, H, W = 2, 512, 32, 32
        features = torch.randn(B, C, H, W)
        
        cd_logits, enhanced_features = change_head(features, mode='scd', return_enhanced=True)
        print_test("SCD forward pass", "PASS")
        
        # Check SCD output
        expected_classes = 49  # 7x7 transitions
        assert cd_logits.shape == (B, expected_classes, 256, 256), f"SCD shape mismatch: {cd_logits.shape}"
        print_test("SCD output shape", "PASS", f"cd_logits: {tuple(cd_logits.shape)}")
        
        # Test BCD mode (LEVIR-MCI)
        cd_logits, enhanced_features = change_head(features, mode='bcd', return_enhanced=True)
        print_test("BCD forward pass", "PASS")
        
        # Check BCD output
        expected_classes = 3
        assert cd_logits.shape == (B, expected_classes, 256, 256), f"BCD shape mismatch: {cd_logits.shape}"
        print_test("BCD output shape", "PASS", f"cd_logits: {tuple(cd_logits.shape)}")
        
        # Check enhanced features
        assert enhanced_features is not None, "Enhanced features are None"
        assert enhanced_features.shape == features.shape, f"Enhanced features shape mismatch"
        print_test("Feature enhancement", "PASS", f"enhanced_features: {tuple(enhanced_features.shape)}")
        
        return True
    except Exception as e:
        print_test("Change Head test", "FAIL", str(e))
        return False


def test_caption_decoder():
    """Test Caption Decoder"""
    print_section("Testing Caption Decoder")
    
    try:
        # Create decoder
        decoder = ChangeGuidedCaptionDecoder(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=3,
            num_change_classes=49,
            max_length=50
        )
        print_test("Caption Decoder initialization", "PASS")
        
        # Test training forward (with teacher forcing)
        B, C, H, W = 2, 512, 32, 32
        T = 50
        visual_features = torch.randn(B, C, H, W)
        change_map = torch.randn(B, 49, 256, 256)
        captions = torch.randint(0, 1000, (B, T))
        lengths = torch.tensor([30, 25])
        
        logits = decoder(visual_features, change_map, captions, lengths, teacher_forcing=True)
        print_test("Training forward pass", "PASS")
        
        # Check output shape
        assert logits.shape == (B, T, 1000), f"Shape mismatch: {logits.shape}"
        print_test("Training output shape", "PASS", f"logits: {tuple(logits.shape)}")
        
        # Test inference (autoregressive generation)
        generated = decoder(visual_features, change_map, captions=None, teacher_forcing=False)
        print_test("Inference forward pass", "PASS")
        
        # Check generated shape
        assert generated.shape[0] == B, f"Batch size mismatch: {generated.shape[0]}"
        assert generated.shape[1] <= 50, f"Generated length too long: {generated.shape[1]}"
        print_test("Inference output shape", "PASS", f"generated: {tuple(generated.shape)}")
        
        return True
    except Exception as e:
        print_test("Caption Decoder test", "FAIL", str(e))
        return False


def test_lsp():
    """Test Learnable Semantic Prompts"""
    print_section("Testing LSP")
    
    try:
        # Test SECOND-CC
        lsp_scd = LearnableSemanticPrompts(
            dataset="second_cc",
            prompt_dim=512,
            learnable=True
        )
        print_test("LSP (SECOND-CC) initialization", "PASS")
        
        # Get prompts
        prompts = lsp_scd.get_transition_prompts()
        assert prompts.shape == (49, 512), f"SECOND-CC prompt shape mismatch: {prompts.shape}"
        print_test("SECOND-CC prompts", "PASS", f"shape: {tuple(prompts.shape)}")
        
        # Test LEVIR-MCI
        lsp_bcd = LearnableSemanticPrompts(
            dataset="levir_mci",
            prompt_dim=512,
            learnable=True
        )
        print_test("LSP (LEVIR-MCI) initialization", "PASS")
        
        prompts = lsp_bcd.get_transition_prompts()
        assert prompts.shape == (3, 512), f"LEVIR-MCI prompt shape mismatch: {prompts.shape}"
        print_test("LEVIR-MCI prompts", "PASS", f"shape: {tuple(prompts.shape)}")
        
        return True
    except Exception as e:
        print_test("LSP test", "FAIL", str(e))
        return False


# =============================================================================
# Full Model Tests
# =============================================================================

def test_full_model_second_cc():
    """Test full UniSCC model for SECOND-CC"""
    print_section("Testing Full Model (SECOND-CC)")
    
    try:
        # Create config
        config = UniSCCConfig(
            dataset='second_cc',
            backbone='swin_base_patch4_window7_224',
            pretrained=False,
            feature_dim=512,
            vocab_size=1000,
            scd_classes=7,
            max_caption_length=50
        )
        print_test("Config creation", "PASS")
        
        # Create model
        model = UniSCC(config)
        print_test("Model initialization", "PASS")
        
        # Count parameters
        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)
        print_test("Parameter count", "PASS", 
                   f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Test training forward
        B, H, W = 2, 256, 256
        img_t0 = torch.randn(B, 3, H, W)
        img_t1 = torch.randn(B, 3, H, W)
        captions = torch.randint(0, 1000, (B, 50))
        lengths = torch.tensor([30, 25])
        
        model.train()
        outputs = model(img_t0, img_t1, captions, lengths)
        print_test("Training forward pass", "PASS")
        
        # Check outputs
        assert 'cd_logits' in outputs, "Missing cd_logits"
        assert 'caption_logits' in outputs, "Missing caption_logits"
        print_test("Output keys", "PASS")
        
        cd_logits = outputs['cd_logits']
        caption_logits = outputs['caption_logits']
        
        # Check shapes
        assert cd_logits.shape == (B, 49, H, W), f"CD shape mismatch: {cd_logits.shape}"
        assert caption_logits.shape == (B, 50, 1000), f"Caption shape mismatch: {caption_logits.shape}"
        print_test("Output shapes", "PASS", 
                   f"cd_logits: {tuple(cd_logits.shape)}, caption_logits: {tuple(caption_logits.shape)}")
        
        # Test inference
        model.eval()
        outputs = model(img_t0, img_t1)
        print_test("Inference forward pass", "PASS")
        
        assert 'generated_captions' in outputs, "Missing generated_captions"
        generated = outputs['generated_captions']
        assert generated.shape[0] == B, f"Batch size mismatch: {generated.shape[0]}"
        print_test("Generation", "PASS", f"generated: {tuple(generated.shape)}")
        
        return True
    except Exception as e:
        print_test("Full model (SECOND-CC) test", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_full_model_levir_mci():
    """Test full UniSCC model for LEVIR-MCI"""
    print_section("Testing Full Model (LEVIR-MCI)")
    
    try:
        # Create config
        config = UniSCCConfig(
            dataset='levir_mci',
            backbone='swin_base_patch4_window7_224',
            pretrained=False,
            feature_dim=512,
            vocab_size=1000,
            bcd_classes=3,
            max_caption_length=50
        )
        print_test("Config creation", "PASS")
        
        # Create model
        model = UniSCC(config)
        print_test("Model initialization", "PASS")
        
        # Test forward
        B, H, W = 2, 256, 256
        img_t0 = torch.randn(B, 3, H, W)
        img_t1 = torch.randn(B, 3, H, W)
        captions = torch.randint(0, 1000, (B, 50))
        lengths = torch.tensor([30, 25])
        
        model.train()
        outputs = model(img_t0, img_t1, captions, lengths)
        print_test("Forward pass", "PASS")
        
        # Check CD output for binary classification
        cd_logits = outputs['cd_logits']
        assert cd_logits.shape == (B, 3, H, W), f"CD shape mismatch: {cd_logits.shape}"
        print_test("BCD output shape", "PASS", f"cd_logits: {tuple(cd_logits.shape)}")
        
        return True
    except Exception as e:
        print_test("Full model (LEVIR-MCI) test", "FAIL", str(e))
        return False


def test_build_function():
    """Test model building from dict"""
    print_section("Testing build_uniscc Function")
    
    try:
        # Build from dict
        config_dict = {
            'dataset': 'second_cc',
            'feature_dim': 512,
            'vocab_size': 1000,
            'pretrained': False
        }
        
        model = build_uniscc(config_dict)
        print_test("Build from dict", "PASS")
        
        # Verify it works
        B, H, W = 2, 256, 256
        img_t0 = torch.randn(B, 3, H, W)
        img_t1 = torch.randn(B, 3, H, W)
        
        outputs = model(img_t0, img_t1)
        print_test("Forward pass after build", "PASS")
        
        return True
    except Exception as e:
        print_test("build_uniscc test", "FAIL", str(e))
        return False


# =============================================================================
# Integration Tests
# =============================================================================

def test_gradient_flow():
    """Test that gradients flow properly"""
    print_section("Testing Gradient Flow")
    
    try:
        config = UniSCCConfig(
            dataset='second_cc',
            pretrained=False,
            vocab_size=1000
        )
        model = UniSCC(config)
        
        # Forward pass
        B = 2
        img_t0 = torch.randn(B, 3, 256, 256, requires_grad=True)
        img_t1 = torch.randn(B, 3, 256, 256, requires_grad=True)
        captions = torch.randint(0, 1000, (B, 50))
        lengths = torch.tensor([30, 25])
        
        outputs = model(img_t0, img_t1, captions, lengths)
        
        # Compute dummy loss
        cd_loss = outputs['cd_logits'].mean()
        cap_loss = outputs['caption_logits'].mean()
        loss = cd_loss + cap_loss
        
        # Backward
        loss.backward()
        print_test("Backward pass", "PASS")
        
        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"
        print_test("Gradient existence", "PASS")
        
        return True
    except Exception as e:
        print_test("Gradient flow test", "FAIL", str(e))
        return False


def test_memory_efficiency():
    """Test memory usage (basic check)"""
    print_section("Testing Memory Efficiency")
    
    try:
        config = UniSCCConfig(
            dataset='second_cc',
            pretrained=False,
            vocab_size=1000,
            feature_dim=512,  # Keep at 512 for compatibility
            hidden_channels=128,  # Smaller hidden channels
            decoder_layers=3,
            tdt_layers=2  # Fewer TDT layers
        )
        model = UniSCC(config)
        
        # Small batch
        B = 1
        img_t0 = torch.randn(B, 3, 256, 256)
        img_t1 = torch.randn(B, 3, 256, 256)
        
        outputs = model(img_t0, img_t1)
        print_test("Small batch forward", "PASS")
        
        # Larger batch
        B = 4
        img_t0 = torch.randn(B, 3, 256, 256)
        img_t1 = torch.randn(B, 3, 256, 256)
        
        outputs = model(img_t0, img_t1)
        print_test("Larger batch forward", "PASS")
        
        return True
    except Exception as e:
        print_test("Memory efficiency test", "FAIL", str(e))
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_checks(verbose: bool = True) -> Dict[str, bool]:
    """
    Run all sanity checks
    
    Args:
        verbose: Print detailed output
    
    Returns:
        Dictionary of test results
    """
    if verbose:
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║         UniSCC Model Sanity Check Suite v1.0              ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
    
    results = {}
    
    # Component tests
    results['encoder'] = test_encoder()
    results['tdt'] = test_tdt()
    results['change_head'] = test_change_head()
    results['caption_decoder'] = test_caption_decoder()
    results['lsp'] = test_lsp()
    
    # Full model tests
    results['full_model_second_cc'] = test_full_model_second_cc()
    results['full_model_levir_mci'] = test_full_model_levir_mci()
    results['build_function'] = test_build_function()
    
    # Integration tests
    results['gradient_flow'] = test_gradient_flow()
    results['memory_efficiency'] = test_memory_efficiency()
    
    # Summary
    if verbose:
        print_section("Test Summary")
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "PASS" if passed_test else "FAIL"
            print_test(test_name.replace('_', ' ').title(), status)
        
        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.ENDC}")
        
        if passed == total:
            print(f"{Colors.OKGREEN}{Colors.BOLD}✓ All tests passed! Model is ready for training.{Colors.ENDC}\n")
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}✗ Some tests failed. Please fix before training.{Colors.ENDC}\n")
    
    return results


def test_component(component_name: str):
    """
    Test a specific component
    
    Args:
        component_name: One of 'encoder', 'tdt', 'change_head', 'caption_decoder', 'lsp'
    """
    tests = {
        'encoder': test_encoder,
        'tdt': test_tdt,
        'change_head': test_change_head,
        'caption_decoder': test_caption_decoder,
        'lsp': test_lsp,
    }
    
    if component_name not in tests:
        print(f"Unknown component: {component_name}")
        print(f"Available: {', '.join(tests.keys())}")
        return False
    
    return tests[component_name]()


if __name__ == "__main__":
    # Run all checks when executed directly
    results = run_all_checks(verbose=True)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
