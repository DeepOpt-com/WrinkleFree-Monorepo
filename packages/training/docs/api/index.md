# API Documentation

Core Python API for WrinkleFree Training.

## Lightning Module

- `wrinklefree.lightning.module.WrinkleFreeLightningModule`: Main Lightning module
- `wrinklefree.lightning.datamodule.WrinkleFreeDataModule`: Data module wrapper
- `wrinklefree.lightning.callbacks`: Training callbacks (GCS, ZClip, TokenCount, etc.)

## Objectives

- `wrinklefree.objectives.manager.ObjectiveManager`: Multi-task objective manager
- `wrinklefree.objectives.factory.create_objective_manager`: Factory function
- `wrinklefree.objectives.continue_pretrain.ContinuePretrainObjective`: CE loss
- `wrinklefree.objectives.dlm.DLMObjective`: Diffusion Language Model masking
- `wrinklefree.objectives.layerwise.LayerwiseObjective`: Hidden state alignment
- `wrinklefree.objectives.logits_distill.LogitsDistillObjective`: KL on teacher logits
- `wrinklefree.objectives.attention_distill.AttentionDistillObjective`: Attention matching
- `wrinklefree.objectives.bitdistill.BitDistillObjective`: Combined distillation
- `wrinklefree.objectives.lrc_reconstruction.LRCReconstructionObjective`: Low-rank correction

## Models

- `wrinklefree.models.bitlinear.BitLinear`: Quantized linear layer with STE
- `wrinklefree.models.subln.SubLN`: Sub-Layer Normalization

## Training Utilities

- `wrinklefree.training.auto_setup.auto_setup_model`: Auto checkpoint resolution + BitNet conversion
- `wrinklefree.training.fsdp_wrapper`: FSDP wrapping with activation checkpointing

## Quantization

- `wrinklefree.quantization.weight_quant`: Ternary weight quantization
- `wrinklefree.quantization.activation_quant`: 8-bit activation quantization
- `wrinklefree.quantization.ste`: Straight-through estimator

## Teachers (for distillation)

- `wrinklefree.teachers.local_teacher.LocalTeacher`: Local HuggingFace teacher
- `wrinklefree.teachers.vllm_teacher.VLLMTeacher`: vLLM-based teacher

## Serving

- `wrinklefree.serving.converter`: GGUF conversion utilities
- `wrinklefree.serving.bitnet_wrapper`: BitNet.cpp Python wrapper

## Data (via data_handler package)

- `data_handler.influence.InfluenceTracker`: Influence-based data remixing
- `data_handler.data.MixedDataset`: Multi-source dataset with dynamic weights
