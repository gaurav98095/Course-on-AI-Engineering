# Examples

## Example 1

Topic: Measuring TTFT and TPOT

Output:

- Warmed-up generation loop with `torch.cuda.synchronize()` around timers
- Streamer-based first-token timestamp
- Explanation of why the first unwarmed run must be discarded
- Debugging note about CUDA async execution making naive `time.time()` lie

## Example 2

Topic: Quantizing the course model with AWQ

Output:

- Calibration data loading
- Quantization config and conversion call
- Before/after memory and tokens/sec table
- Explanation of which layers stay in higher precision and why

## Example 3

Topic: A LitServe endpoint

Output:

- `LitAPI` subclass with `setup`, `decode_request`, `predict`, `encode_response`
- Model loaded once in `setup`, never per request
- Logging tokens/sec per request
- Debugging note about batch timeout settings
