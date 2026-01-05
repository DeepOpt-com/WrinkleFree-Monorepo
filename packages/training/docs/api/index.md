# API Documentation

Core Python API for WrinkleFree Training.

## Lightning Module

- `wf_train.lightning.module.WrinkleFreeLightningModule`: Main Lightning module
- `wf_train.lightning.datamodule.WrinkleFreeDataModule`: Data module wrapper
- `wf_train.lightning.callbacks`: Training callbacks (GCS, ZClip, TokenCount, etc.)

## Objectives

- `wf_train.objectives.manager.ObjectiveManager`: Multi-task objective manager
- `wf_train.objectives.factory.create_objective_manager`: Factory function
- `wf_train.objectives.continue_pretrain.ContinuePretrainObjective`: CE loss
- `wf_train.objectives.dlm.DLMObjective`: Diffusion Language Model masking
- `wf_train.objectives.layerwise.LayerwiseObjective`: Hidden state alignment
- `wf_train.objectives.logits_distill.LogitsDistillObjective`: KL on teacher logits
- `wf_train.objectives.attention_distill.AttentionDistillObjective`: Attention matching
- `wf_train.objectives.bitdistill.BitDistillObjective`: Combined distillation
- `wf_train.objectives.lrc_reconstruction.LRCReconstructionObjective`: Low-rank correction

## Models

- `wf_train.models.bitlinear.BitLinear`: Quantized linear layer with STE
- `wf_train.models.subln.SubLN`: Sub-Layer Normalization

## Training Utilities

- `wf_train.training.auto_setup.auto_setup_model`: Auto checkpoint resolution + BitNet conversion
- `wf_train.training.fsdp_wrapper`: FSDP wrapping with activation checkpointing

## Quantization

- `wf_train.quantization.weight_quant`: Ternary weight quantization
- `wf_train.quantization.activation_quant`: 8-bit activation quantization
- `wf_train.quantization.ste`: Straight-through estimator

## Teachers (for distillation)

- `wf_train.teachers.local_teacher.LocalTeacher`: Local HuggingFace teacher
- `wf_train.teachers.vllm_teacher.VLLMTeacher`: vLLM-based teacher

## Serving

- `wf_train.serving.converter`: GGUF conversion utilities
- `wf_train.serving.bitnet_wrapper`: BitNet.cpp Python wrapper

## Data (via wf_data package)

- `wf_data.influence.InfluenceTracker`: Influence-based data remixing
- `wf_data.data.MixedDataset`: Multi-source dataset with dynamic weights
