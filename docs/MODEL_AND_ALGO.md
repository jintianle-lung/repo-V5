# R5 Detection-Gated Residual Inversion

The released algorithm treats tactile nodule characterization as a gated inverse problem.

1. A CNN-MSTCN evidence detector processes all 10-frame tactile windows and predicts whether a window contains lesion-responsive evidence.
2. The detector threshold is selected on validation data and then frozen.
3. A residual inversion branch receives the frozen detector feature, detector probability, and a trainable morphology feature from the same tactile sequence.
4. Size is the main endpoint. The continuous size estimate is the expected physical size from the seven-class size distribution plus a bounded residual correction.
5. Depth is an auxiliary endpoint conditioned on the seven-class size route, coarse-size route, and expected size.

This formulation is intentionally conservative: detection defines the reliable evidence boundary, size is the dominant recoverable inverse endpoint, and depth is reported as a conditional residual signal.

