# PyTorch DataLoader Integration Plan for Scangen

## Executive Summary
Extend scangen to provide a PyTorch Dataset/DataLoader interface that generates RAW data on-the-fly from RGB datasets during training, while maintaining backward compatibility with the existing batch processing CLI.

## Architecture Overview

### Current Architecture
```
CLI → RAWGenerator → Batch Processing → Save to Disk (PNG/PKL)
```

### Proposed Addition
```
RGB Dataset → RawDataset Wrapper → DataLoader → Training Loop
                    ↓
              RAWGenerator (reused)
                    ↓
              Real-time RAW Generation
```

## Implementation Plan

### Phase 1: Core Dataset Implementation

#### Step 1.1: Create Base RawDataset Class
**File**: `src/scangen/data/raw_dataset.py`

**Requirements**:
- Inherit from `torch.utils.data.Dataset`
- Accept any RGB dataset that follows PyTorch Dataset interface
- Store reference to RAWGenerator instance
- Configure noise model (DND/SIDD/custom)

**Definition of Done**:
- [ ] RawDataset class created with `__init__`, `__len__`, `__getitem__`
- [ ] Can wrap CIFAR-10 dataset
- [ ] Returns (rgb_image, clean_raw, noisy_raw, metadata) tuples
- [ ] Unit test passes for basic dataset operations

#### Step 1.2: Implement Transform Pipeline
**File**: `src/scangen/data/raw_dataset.py`

**Requirements**:
- Handle different input image sizes (32x32 CIFAR, 224x224 ImageNet, etc.)
- Apply necessary preprocessing (resize, normalize)
- Support configurable target RAW size

**Definition of Done**:
- [ ] Correctly handles CIFAR-10 (32x32 → 256x256)
- [ ] Configurable target_size parameter works
- [ ] Unit tests for different input sizes pass

#### Step 1.3: Optimize Memory and Performance
**File**: `src/scangen/data/raw_dataset.py`

**Requirements**:
- Lazy loading of RGB2RAW model weights
- Efficient tensor operations (no unnecessary copies)
- Optional caching for repeated access

**Definition of Done**:
- [ ] Model loads only once on first access
- [ ] Memory profiling shows no leaks
- [ ] Performance benchmark: >100 samples/second on GPU
- [ ] Optional caching mechanism implemented and tested

### Phase 2: DataLoader Factory and Utilities

#### Step 2.1: Create DataLoader Factory
**File**: `src/scangen/data/raw_dataloader.py`

**Requirements**:
- Factory function `create_raw_dataloader()`
- Handle device placement (CPU/CUDA/MPS)
- Configure batch size, num_workers, pin_memory
- Support distributed training setup

**Definition of Done**:
- [ ] Factory function creates working DataLoader
- [ ] Correct device placement for different backends
- [ ] Multi-worker loading works without errors
- [ ] Unit test with actual training step passes

#### Step 2.2: Add Dataset Registry
**File**: `src/scangen/data/dataset_registry.py`

**Requirements**:
- Pre-configured loaders for common datasets
- Support: CIFAR-10, CIFAR-100, ImageNet, custom datasets
- Automatic download option for public datasets

**Definition of Done**:
- [ ] Registry with at least 3 dataset configurations
- [ ] `get_dataset()` function works for all registered datasets
- [ ] Automatic download works for CIFAR datasets
- [ ] Documentation for adding custom datasets

#### Step 2.3: Implement Noise Configuration Interface
**File**: `src/scangen/data/noise_config.py`

**Requirements**:
- Structured noise configuration classes
- Support DND, SIDD, and custom noise models
- Per-batch or per-epoch noise variation options
- Deterministic mode for reproducibility

**Definition of Done**:
- [ ] NoiseConfig base class and implementations
- [ ] Can switch noise models during training
- [ ] Reproducible results with seed setting
- [ ] Unit tests for all noise configurations

### Phase 3: Integration and Compatibility

#### Step 3.1: Update Package Exports
**File**: `src/scangen/__init__.py` and `src/scangen/data/__init__.py`

**Requirements**:
- Export new Dataset and DataLoader classes
- Maintain backward compatibility
- Clear import paths

**Definition of Done**:
- [ ] Can import: `from scangen.data import RawDataset, create_raw_dataloader`
- [ ] Existing CLI commands still work
- [ ] No breaking changes to current API
- [ ] Import tests pass

#### Step 3.2: Add Streaming Mode to RAWGenerator
**File**: `src/scangen/pipeline/generator.py`

**Requirements**:
- Add streaming flag to avoid unnecessary tensor operations
- Optimize for single-sample generation (for DataLoader)
- Maintain batch generation for CLI

**Definition of Done**:
- [ ] RAWGenerator supports both batch and streaming modes
- [ ] No performance regression in batch mode
- [ ] Streaming mode optimized for single samples
- [ ] Unit tests for both modes

#### Step 3.3: Create Usage Examples
**File**: `examples/dataloader_training.py`

**Requirements**:
- Complete training example with denoising model
- Show integration with popular frameworks (PyTorch Lightning, etc.)
- Demonstrate different dataset configurations

**Definition of Done**:
- [ ] Working training script for image denoising
- [ ] Example runs without errors on CPU and GPU
- [ ] Clear comments and documentation
- [ ] Can reproduce training results

### Phase 4: Testing and Documentation

#### Step 4.1: Comprehensive Unit Tests
**File**: `tests/test_raw_dataset.py`, `tests/test_raw_dataloader.py`

**Requirements**:
- Test all dataset configurations
- Test edge cases (empty dataset, single sample, etc.)
- Test memory efficiency
- Test multi-processing compatibility

**Definition of Done**:
- [ ] >90% code coverage for new modules
- [ ] All tests pass on CI/CD
- [ ] Memory leak tests pass
- [ ] Concurrency tests pass

#### Step 4.2: Integration Tests
**File**: `tests/test_integration_dataloader.py`

**Requirements**:
- End-to-end test with actual training loop
- Test with different datasets
- Test with different hardware (CPU, CUDA if available)
- Performance benchmarks

**Definition of Done**:
- [ ] Full training loop test passes
- [ ] Works with at least 3 different datasets
- [ ] Performance meets targets (>100 samples/sec on GPU)
- [ ] No memory leaks during extended training

#### Step 4.3: Documentation
**Files**: `docs/dataloader_usage.md`, API documentation

**Requirements**:
- Usage guide for DataLoader interface
- API reference documentation
- Migration guide for existing users
- Performance tuning guide

**Definition of Done**:
- [ ] Complete usage documentation
- [ ] API docs generated and readable
- [ ] Examples for common use cases
- [ ] Troubleshooting section

### Phase 5: Demonstration and Validation

#### Step 5.1: Create Comprehensive Demo
**File**: `tests/test_pytorch_pipeline_demo.py`

This will be the final validation test demonstrating the complete pipeline:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import scangen.data as sdata

def test_complete_pytorch_pipeline():
    """
    Comprehensive test demonstrating full PyTorch training pipeline
    using scangen as a DataLoader for RAW data generation.
    """
    # 1. Load base RGB dataset
    rgb_dataset = CIFAR10(root='./data', train=True, download=True)
    
    # 2. Wrap with RawDataset
    raw_dataset = sdata.RawDataset(
        rgb_dataset=rgb_dataset,
        model_path='data/weights/rgb2raw.pth',
        noise_config={'type': 'dnd'},
        target_size=(256, 256),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 3. Create DataLoader
    dataloader = DataLoader(
        raw_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 4. Define simple denoising model
    model = nn.Sequential(
        nn.Conv2d(4, 64, 3, padding=1),  # 4 channels for RGGB
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 4, 3, padding=1),
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 5. Training step
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for batch_idx, (rgb, clean_raw, noisy_raw, metadata) in enumerate(dataloader):
        if batch_idx >= 5:  # Just test a few batches
            break
            
        # Forward pass
        optimizer.zero_grad()
        denoised = model(noisy_raw)
        loss = criterion(denoised, clean_raw)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Assertions
        assert rgb.shape[0] == 32  # Batch size
        assert clean_raw.shape == noisy_raw.shape
        assert clean_raw.shape[1] == 4  # RGGB channels
        assert loss.item() > 0
        
    return True
```

**Definition of Done**:
- [ ] Demo script runs successfully
- [ ] Produces valid RAW data
- [ ] Training loop converges
- [ ] Memory usage is stable
- [ ] Performance metrics logged

## Success Criteria

### Functional Requirements
- [ ] Can import and use as PyTorch DataLoader
- [ ] Generates RAW data on-the-fly without disk I/O
- [ ] Supports multiple input datasets (CIFAR, ImageNet, custom)
- [ ] Maintains backward compatibility with CLI

### Performance Requirements  
- [ ] Generation speed: >100 samples/second on GPU
- [ ] Memory efficient: <2GB overhead for DataLoader
- [ ] Multi-worker support without bottlenecks

### Quality Requirements
- [ ] >90% test coverage for new code
- [ ] No memory leaks
- [ ] Thread-safe for multi-worker loading
- [ ] Reproducible with seed setting

## Timeline Estimate

- Phase 1: 2-3 days (Core Dataset Implementation)
- Phase 2: 2 days (DataLoader Factory and Utilities)
- Phase 3: 1-2 days (Integration and Compatibility)
- Phase 4: 2 days (Testing and Documentation)
- Phase 5: 1 day (Demonstration and Validation)

**Total: 8-10 days**

## Risk Mitigation

### Risk 1: Performance Bottleneck
**Mitigation**: Profile early, optimize tensor operations, consider caching strategies

### Risk 2: Memory Issues with Large Datasets
**Mitigation**: Implement lazy loading, use memory mapping for large datasets

### Risk 3: Multi-processing Complications  
**Mitigation**: Careful handling of model initialization, use proper worker init functions

### Risk 4: Device Compatibility
**Mitigation**: Test on CPU, CUDA, and MPS backends early in development

## Next Steps

1. Review and refine this plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Regular testing and validation at each phase
5. Documentation as we go

---

**Note**: This plan is designed to be iterative. After review and approval, each phase can be adjusted based on discoveries during implementation.

## Functionality Breakdown: Core vs Optional

### CORE Functionality (Minimum Viable Product)

These features are essential for the DataLoader to be functional and useful:

#### 1. Basic Dataset Wrapper (Phase 1, Step 1.1)
- **What**: `RawDataset` class that wraps RGB datasets
- **Why Critical**: Foundation of entire DataLoader functionality
- **Deliverable**: Working `__getitem__` that returns RAW data
- **Estimated Time**: 1 day

#### 2. RGB-to-RAW Pipeline Integration (Phase 1, Step 1.1-1.2)
- **What**: Use existing RAWGenerator for conversions
- **Why Critical**: Core transformation functionality
- **Deliverable**: Successful RGB → RAW conversion in DataLoader
- **Estimated Time**: 1 day

#### 3. Basic Noise Support (Phase 1, Step 1.1)
- **What**: Support for DND and SIDD noise models
- **Why Critical**: RAW data without noise isn't realistic
- **Deliverable**: Can apply existing noise models
- **Estimated Time**: 0.5 days

#### 4. CIFAR-10 Support (Phase 1, Step 1.2)
- **What**: Handle CIFAR-10 dataset specifically
- **Why Critical**: Primary use case, proof of concept
- **Deliverable**: Working with CIFAR-10 end-to-end
- **Estimated Time**: 0.5 days

#### 5. Basic DataLoader Creation (Phase 2, Step 2.1)
- **What**: Simple factory function or direct DataLoader usage
- **Why Critical**: Standard PyTorch integration
- **Deliverable**: Can create working DataLoader
- **Estimated Time**: 0.5 days

#### 6. Single Worker Support (Phase 3, Step 3.1)
- **What**: Works with num_workers=0
- **Why Critical**: Must work in simplest case
- **Deliverable**: No crashes with single-threaded loading
- **Estimated Time**: 0.5 days

#### 7. Basic Test Coverage (Phase 4, Step 4.1)
- **What**: Unit tests for core functionality
- **Why Critical**: Ensure reliability
- **Deliverable**: Tests for basic dataset operations
- **Estimated Time**: 1 day

#### 8. Minimal Documentation (Phase 4, Step 4.3)
- **What**: Basic usage examples and docstrings
- **Why Critical**: Users need to know how to use it
- **Deliverable**: README section with example usage
- **Estimated Time**: 0.5 days

**Total CORE Implementation Time: 5.5 days**

### OPTIONAL Functionality (Enhancements)

These features improve usability, performance, and flexibility:

#### 1. Performance Optimizations (Phase 1, Step 1.3)
- **What**: Caching, lazy loading, tensor optimization
- **Why Optional**: Works without these, just slower
- **Priority**: HIGH - Significant user experience improvement
- **Estimated Time**: 1.5 days

#### 2. Multi-Dataset Support (Phase 2, Step 2.2)
- **What**: ImageNet, CIFAR-100, custom datasets
- **Why Optional**: CIFAR-10 proves concept
- **Priority**: HIGH - Broadens applicability
- **Estimated Time**: 1 day

#### 3. Dataset Registry (Phase 2, Step 2.2)
- **What**: Pre-configured dataset setups
- **Why Optional**: Users can configure manually
- **Priority**: MEDIUM - Convenience feature
- **Estimated Time**: 1 day

#### 4. Advanced Noise Configuration (Phase 2, Step 2.3)
- **What**: Per-batch variation, custom noise models
- **Why Optional**: Basic noise models sufficient for most cases
- **Priority**: MEDIUM - Research flexibility
- **Estimated Time**: 1 day

#### 5. Multi-Worker Support (Phase 2, Step 2.1)
- **What**: Efficient parallel data loading
- **Why Optional**: Works with single worker
- **Priority**: HIGH - Major performance improvement
- **Estimated Time**: 1 day

#### 6. Distributed Training Support (Phase 2, Step 2.1)
- **What**: DDP compatibility, proper sampling
- **Why Optional**: Single-GPU training works
- **Priority**: LOW - Advanced use case
- **Estimated Time**: 0.5 days

#### 7. Streaming Mode Optimization (Phase 3, Step 3.2)
- **What**: Separate code path for single samples
- **Why Optional**: Batch mode works for DataLoader
- **Priority**: LOW - Minor optimization
- **Estimated Time**: 1 day

#### 8. Framework Integrations (Phase 3, Step 3.3)
- **What**: PyTorch Lightning, HuggingFace examples
- **Why Optional**: Standard PyTorch is sufficient
- **Priority**: MEDIUM - User convenience
- **Estimated Time**: 1 day

#### 9. Comprehensive Testing (Phase 4, Step 4.2)
- **What**: Edge cases, memory profiling, benchmarks
- **Why Optional**: Basic tests ensure functionality
- **Priority**: MEDIUM - Quality assurance
- **Estimated Time**: 1 day

#### 10. Advanced Documentation (Phase 4, Step 4.3)
- **What**: Performance tuning guide, troubleshooting
- **Why Optional**: Basic docs sufficient to start
- **Priority**: LOW - Nice to have
- **Estimated Time**: 0.5 days

**Total OPTIONAL Implementation Time: 9.5 days**

### Implementation Strategy

#### Minimum Viable Product (MVP) Path
1. **Week 1**: Complete all CORE functionality (5.5 days)
2. **Testing**: Validate MVP works end-to-end
3. **Release**: v0.2.0 with basic DataLoader support

#### Full Feature Path
1. **Week 1**: CORE functionality
2. **Week 2**: HIGH priority optional features
   - Performance optimizations
   - Multi-dataset support
   - Multi-worker support
3. **Week 3**: MEDIUM priority features
   - Dataset registry
   - Advanced noise configuration
   - Framework integrations
   - Comprehensive testing

### Success Metrics for MVP

**Must Have**:
- [ ] CIFAR-10 DataLoader works
- [ ] Returns (rgb, clean_raw, noisy_raw) tuples
- [ ] No crashes with batch_size=32, num_workers=0
- [ ] Basic example in documentation runs
- [ ] >80% test coverage on core modules

**Performance Targets (Relaxed for MVP)**:
- [ ] >30 samples/second on GPU (vs 100 for full)
- [ ] <4GB memory overhead (vs 2GB for full)
- [ ] Works with batch sizes 1-128

### Decision Points

After MVP implementation, evaluate:

1. **Performance**: Is 30 samples/sec sufficient for users?
   - If NO → Prioritize performance optimizations
   - If YES → Move to multi-dataset support

2. **User Feedback**: What features are most requested?
   - Multi-worker support?
   - More datasets?
   - Better documentation?

3. **Use Cases**: How are people using it?
   - Research → Prioritize flexibility features
   - Production → Prioritize performance/stability
   - Education → Prioritize documentation/examples

This breakdown allows for incremental delivery with a functional product available much sooner than the full implementation.