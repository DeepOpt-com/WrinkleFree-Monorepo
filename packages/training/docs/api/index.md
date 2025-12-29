# API Documentation

This section documents the core Python API for WrinkleFree-1.58Quant.

## Models

- `wrinklefree.models.bitlinear.BitLinear`: Quantized linear layer with STE.
- `wrinklefree.models.subln.SubLN`: Sub-Layer Normalization module.
- `wrinklefree.models.llama.LlamaForCausalLM`: Main model architecture.

## Training

- `wrinklefree.training.trainer.Trainer`: Base trainer class.
- `wrinklefree.training.stage1.run_stage1`: SubLN insertion (Stage 1).
- `wrinklefree.training.stage1_9.Stage19Trainer`: Layer-wise distillation trainer (Stage 1.9).
- `wrinklefree.training.stage1_9.HiddenStateTeacherWrapper`: Teacher model wrapper for hidden state extraction.
- `wrinklefree.training.stage1_9.run_stage1_9`: Run Stage 1.9 layer-wise distillation.
- `wrinklefree.training.stage2.Stage2Trainer`: Continue pre-training trainer (Stage 2).
- `wrinklefree.training.stage3.Stage3Trainer`: Distillation trainer (Stage 3).

## Distillation

- `wrinklefree.distillation.combined_loss.BitDistillLoss`: Combined loss function (CE + logits KL + attention KL).
- `wrinklefree.distillation.logits_loss.LogitsDistillationLoss`: Logits KL divergence.
- `wrinklefree.distillation.attention_loss.AttentionDistillationLoss`: Attention distillation.
- `wrinklefree.distillation.layerwise_loss.LayerwiseDistillationLoss`: Layer-wise hidden state distillation (Stage 1.9).
- `wrinklefree.distillation.layerwise_loss.LayerwiseLossType`: Enum for loss metrics (cosine, mse, mse_normalized, kl, inner_product).

## Influence (via Library)

- `cheapertraining.InfluenceAwareOptimizer`: Optimizer wrapper for influence updates.
- `cheapertraining.DataInfCalculator`: Influence score calculator.
- `cheapertraining.MixtureWeightCalculator`: Mixture weight optimizer.
