# Document 6: Failure Mechanism

Embedding-only reranking fails in adversarial settings because
it prioritizes surface-level semantic similarity.

Chunks that repeat domain keywords or fluent explanations
can appear highly relevant in embedding space,
even when they do not contain the actual causal explanation.

Answer-aware reranking mitigates this by:
- first generating a tentative answer
- then evaluating which chunks truly support that answer
- suppressing chunks that only look relevant but do not provide evidence