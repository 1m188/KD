# Missing Attention Distillation Loss in `DistilledPI0Pytorch`

## Summary

It appears that the **attention distillation loss** (described in Section 5.2 of the Shallow-π paper) is not implemented in the current distillation code. The `DistilledPI0Pytorch.forward()` method only computes two of the three distillation objectives described in the paper. I'd like to ask whether this is intentional or an oversight.

## Details

### What the paper describes

The paper (arXiv:2601.20262) states in the contributions (Section 1) that the framework includes **three** distillation objectives:

> "We carefully design and systematically ablate a set of distillation objectives—including **ground-truth supervision**, **teacher trajectory imitation**, and **intermediate attention transfer**—tailored to π-like flow-based VLAs, where only action tokens are denoised and multimodal features are injected from the VLM backbone at every layer."

Additionally, Section 5.2 is specifically dedicated to **"Attention Distillation"**, which describes how the attention maps from the teacher model are used to guide the student model's attention behavior—particularly the cross-attention between the action head and the VLM backbone at every layer.

### What the code implements

In `src/openpi/models_pytorch/pi0_pytorch.py`, the `DistilledPI0Pytorch.forward()` method computes only **two** losses:

```python
loss_output_gt = F.mse_loss(u_t, v_t, reduction="mean")        # ground-truth supervision
loss_output_teacher = F.mse_loss(v_t_teacher, v_t, reduction="mean")  # teacher trajectory imitation

return loss_output_gt + loss_output_teacher
```

The **attention distillation loss** (i.e., aligning the intermediate attention maps between the teacher and student) does not appear to be present in the code. I also checked `scripts/distill_pytorch.py` and did not find any attention map extraction or attention-based loss computation there either.

### Key observations

1. The teacher model's forward pass (`teacher.eval_model(...)`) is called under `torch.no_grad()` and only returns the final velocity prediction `v_t_teacher`—the intermediate attention weights are not captured.
2. The student model's forward pass similarly does not expose or return intermediate attention maps.
3. There is no loss term that compares attention distributions between the teacher and student at corresponding (or mapped) layers.

## Question

Could you clarify whether:

1. The attention distillation loss was intentionally omitted from the released code (e.g., because it was found to have minimal impact in ablations), and the current two-loss setup is sufficient to reproduce the paper's results?
2. Or is this an oversight, and the attention distillation code will be added in a future update?

This would be very helpful for anyone trying to understand or reproduce the full method. Thank you for the great work!

## References

- Paper: [Shallow-π: Knowledge Distillation for Flow-based VLAs](https://arxiv.org/abs/2601.20262) (arXiv:2601.20262)
- Relevant code location: `src/openpi/models_pytorch/pi0_pytorch.py` → `DistilledPI0Pytorch.forward()`
- Distillation script: `scripts/distill_pytorch.py`
