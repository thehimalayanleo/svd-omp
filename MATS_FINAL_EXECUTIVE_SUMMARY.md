# Causal sub-updates that replicate across models and seeds

## Executive summary

Fine-tuning changes millions of model weights at once. Model-diffing methods can often find directions correlated with the new behavior, but correlation is not enough for model forensics. I asked a stricter question: can we identify one concrete part of the learned weight update that both recreates a behavioral regression when added to the base model and repairs it when removed from the fine-tuned model?

I studied a harmless synthetic regression in Mistral Small 3.1 24B. A rank-16 LoRA organism learned to follow an irrelevant note saying that answer A was entered first. The base model correctly answered B, while the post-trained model incorrectly answered A. Because LoRA exposes the exact update, I decomposed each attention-output update into rank-one SVD atoms. Across 40 layers this gave 640 atoms. A selected support was always applied with its original coefficient of one. I did not tune the intervention strength on confirmation.

The causal test was deliberately bidirectional. For each question source, the same sub-update had to:

1. turn the base model's correct B answer into the trained A error;
2. turn the post-trained model's A error back into B when subtracted;
3. preserve a matched A control and nearby protected behaviors.

The strongest result is a prospective Qwen3-30B replication. Before training, I froze five new seeds, 92 sources unused by the earlier Qwen campaign, source-disjoint selection, validation, and confirmation splits, a 272-of-768 atom support, every comparator, a 4/5 campaign gate, and a float32 unmerged full-dictionary endpoint check. All five organisms passed admission and causal validation. On 16 untouched confirmation sources per seed, the same sub-update produced 80/80 bidirectional effects, kept every protected family correct, caused zero paired-control failures, and closed both full-update endpoint directions exactly at the prediction level. The campaign passed 5/5.

Each Qwen support also beat 999 same-size random supports, with zero ties and empirical p=0.001 per seed. FoBa+SVD, OMP+SVD, top-SVD, and cross-seed consensus all reached 80/80. Gradient ranking and direct OMP-272 reached 0/80. The causal effect therefore replicated more strongly than the selector story. I do not claim FoBa or OMP superiority.

Earlier versions failed for useful reasons. One dataset used controls the base model did not support. Another reconstructed a different prompt from the one used during capability screening. A corrected prompt exposed that only two of five organisms reliably learned the regression. After matching the training and evaluation instructions, all five new organisms passed the input gate, but a frozen 64-atom support still failed validation. I kept each failure in the denominator and left confirmation sealed.

Using only the opened development split, I fixed the final support size at 224 of 640 atoms. The selector began with a 64-atom weighted OMP support, applied eight fixed-cardinality FoBa swaps, and filled the remaining budget by singular value. All five exact supports then passed source-disjoint validation, which opened a still-untouched 10-source confirmation split.

Every independent training seed achieved 9/10 bidirectional confirmation successes, for 45/50 total. Every protected family remained 10/10 in both intervention directions, and no matched pair was damaged. The frozen system rule required at least three supports to issue and every issued support to pass. All five issued and all five passed.

The selector comparison prevented a more exciting but false conclusion. At the same 224-atom budget, FoBa plus SVD, OMP plus SVD, and top-SVD each reached 45/50. Gradient ranking reached 48/50, and the full update reached 50/50. At 64 atoms, no method formed a reliable system. Therefore, this is not evidence that FoBa or OMP is the best selector. It is evidence that a structured part of an exact fine-tuning update can remain sufficient, necessary, and behaviorally specific across independent training runs and sealed sources.

The result is bounded. The Qwen support contains 35.4% of the atom dictionary, the organisms learn a synthetic regression, and individual atoms do not yet have stable semantic meanings. The final Qwen campaign is prospective and uses fresh sources, but it repeats the same controlled behavior rather than solving natural-checkpoint model diffing. My next experiment would freeze predictors of causal repairability on new behaviors and run a matched Delta-Crosscoder or related learned model-diffing baseline on a smaller shared organism.

The main lesson is methodological: model-diffing features should be judged by exact interventions, both causal directions, matched controls, and retained failures. A direction that looks important is not yet a causal account of what training changed.

## Second behavior transfer, retained as a boundary result

I then held the 224-atom procedure fixed and moved to five fresh Mistral 24B organisms with a distinct harmless metadata-triggered over-abstention regression. On fresh confirmation sources, all five supports produced 10/10 bidirectional target outcomes, preserved at least 9/10 in every protected family, caused zero paired-control damage, and outperformed almost all 999 same-size random supports per seed. However, the frozen full-dictionary merged-BF16 endpoint cycle missed one of 60 ablation rows in three seeds, so only 2/5 passed the literal protocol and the required 4/5 transfer gate failed. I report this as strong behavioral-transfer evidence and an honest failed general-transfer claim, not as confirmation that the method is universally reliable.
