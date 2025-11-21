# Repetition Bug in Fine-Tuned Whisper Models on Mobile (FUTO)

## The Problem

When converting fine-tuned Whisper models to GGUF format for use on mobile devices (specifically FUTO Voice Input), some models—particularly smaller ones like Whisper Tiny—exhibit a repetition bug where the model enters an infinite loop, repeating the same transcribed text 20-30 times instead of stopping after completing the transcription.

**Example behavior:**
- Input: "I'm going to the shop"
- Expected output: "I'm going to the shop"
- Actual output: "I'm going to the shop I'm going to the shop I'm going to the shop..." (repeating 20-30 times)

## What This Indicates

This repetition behavior suggests several possible issues:

### 1. **End-of-Sequence (EOS) Token Problems**

The most likely cause is that the model's EOS (end-of-sequence) token mechanism is not functioning correctly:

- **During fine-tuning:** If the training data didn't properly include or reinforce EOS token behavior, the model may not have learned when to stop generating output
- **During conversion:** The GGUF conversion process may have incorrectly mapped or lost the EOS token information
- **During inference:** The mobile inference engine may not be properly detecting or respecting the EOS token

### 2. **Quantization Issues**

Converting to GGUF typically involves quantization (reducing precision from FP32/FP16 to INT8 or INT4):

- **Threshold sensitivity:** The stopping criteria in Whisper models rely on probability thresholds. Quantization can alter these probabilities enough that the stopping condition is never met
- **Smaller models more affected:** Whisper Tiny has fewer parameters and less capacity to handle quantization-induced errors compared to larger variants
- **Critical parameters affected:** The specific weights controlling sequence termination may be disproportionately affected by quantization

### 3. **Context Window or Attention Issues**

The conversion or mobile inference may have issues with:

- **Max length parameter:** The maximum generation length may be set incorrectly or ignored
- **Attention mask:** Problems with the attention mechanism could cause the model to lose track of what it has already generated
- **Memory state:** Issues with the model's internal state tracking between chunks

### 4. **Fine-Tuning Artifacts**

The fine-tuning process itself may have introduced problems:

- **Insufficient training steps:** The model may not have converged properly during fine-tuning
- **Learning rate issues:** Too high a learning rate could have destabilized the model's stopping behavior
- **Data imbalance:** If the training data had unusual characteristics (very short or very long samples), the model may have learned incorrect stopping patterns

## Diagnostic Steps

To narrow down the cause:

1. **Test the pre-conversion model:** Use the fine-tuned model on desktop before GGUF conversion. If it works there but not on mobile, the issue is in conversion/mobile inference

2. **Test different quantization levels:** Try converting with different quantization settings (Q8_0 vs Q4_0 vs Q5_1) to see if precision loss is the culprit

3. **Test with different model sizes:** If only Tiny exhibits this behavior, quantization sensitivity is likely the issue

4. **Inspect the conversion logs:** Look for warnings or errors during GGUF conversion, particularly around special tokens

5. **Compare tokenizer outputs:** Verify that the tokenizer is correctly handling special tokens (especially `<|endoftext|>`) in both desktop and mobile environments

## Solutions and Workarounds

### Short-term fixes:

1. **Use a larger model variant:** Try Whisper Base or Small instead of Tiny—they handle quantization better

2. **Use higher quantization precision:** If storage allows, use Q8_0 instead of Q4_0 quantization

3. **Implement external stopping:** Add inference-time maximum token limits or timeout mechanisms in the mobile app

### Long-term fixes:

1. **Improve fine-tuning:** Ensure training data includes proper sequence boundaries and the model is trained to convergence

2. **Add EOS reinforcement:** During fine-tuning, you can add additional training emphasis on EOS token behavior

3. **Test conversion tools:** Different GGUF conversion tools (llama.cpp, ct2-transformers-converter, etc.) may handle the conversion differently

4. **Report to FUTO:** This may be a bug in FUTO's inference engine that needs fixing

## Prevention in Future Fine-Tuning

To avoid this issue in future fine-tuning projects:

1. **Validate before conversion:** Always test fine-tuned models thoroughly on desktop before converting to mobile formats

2. **Include diverse audio lengths:** Ensure training data has samples of various lengths to teach proper stopping behavior

3. **Monitor validation metrics:** Watch for unusual patterns in validation that might indicate stopping behavior issues

4. **Test multiple model sizes:** Fine-tune both Tiny and Base variants to ensure the approach works across model sizes

5. **Document conversion parameters:** Keep detailed records of conversion settings so you can iterate if problems occur

## Additional Context

- **Desktop inference success:** The fact that the model worked correctly on desktop indicates the fine-tuning itself was likely successful
- **Inference was happening:** The model was correctly transcribing the initial phrase, showing that the core model weights were intact
- **Model-specific behavior:** The issue affecting Tiny but potentially not other sizes points to quantization sensitivity

This type of bug is frustrating but common when deploying fine-tuned models to resource-constrained environments. The good news is that inference was occurring correctly—the issue is specifically with sequence termination, which is usually fixable through conversion parameter adjustments or using slightly larger model variants.

---

*Note: This document was generated by Claude Code, an AI assistant. Please validate technical details and test recommendations in your specific environment before implementing.*
