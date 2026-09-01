# Three-minute video transcript

Hi, I am Ajinkya. I studied a simple model-forensics question: when
post-training creates one bad rule, can we find a few pieces of the weight
update that causally implement that rule and remove them without changing
neighboring behavior?

I built a harmless Qwen3-4B model organism. It still answers normal questions,
ignores quoted instructions, and abstains when information is genuinely
missing. But it learns one regression: if a valid question contains a harmless
provenance warning, it abstains instead of answering.

Here is the method. For each attention output layer, I subtract the base weight
from the post-trained weight. I decompose that update into rank-one SVD atoms.
Each atom is one small direction that can be removed during inference. A
source-paired gradient scores an atom by asking: does removing it help the
warning target, while leaving the same source's genuinely unanswerable warning
control and the other protected behaviors unchanged?

I use development data to choose only three or four atoms and one intervention
dose. Then I freeze everything. The final runner contains no search or
calibration code.

The final test uses 24 questions that never appeared in training, development,
or any earlier causal test. Each repair target is paired with a control built
from the same source. A repair only counts if the model answers the valid
warning question and still abstains on the genuinely unanswerable warning
question. This rules out the easy shortcut of simply suppressing abstention.

I preregistered the data hashes, model revision, two fresh training seeds,
supports, doses, thresholds, twenty same-budget random supports per seed, and
the exact pass rule before opening predictions.

Both seeds passed. On seed 349, three selected atoms specifically repaired 12
of 24 targets. On seed 353, four atoms repaired 19 of 24. In both cases there
were zero shortcut repairs, zero paired-control failures, and 24 of 24 correct
decisions on every protected family. The selected support also beat all twenty
matched random supports on both seeds, giving an add-one empirical probability
of 1 in 21 per seed.

The negative results matter. Earlier, activation energy looked like a strong
repair method, but a new factorial control revealed that it was broadly
suppressing valid abstention. An earlier preregistration also stopped before
the causal test because a fresh organism missed the admission threshold. I did
not lower that threshold. I improved the organism recipe, kept the final test
sealed, and repeated the complete protocol on new seeds.

So the claim is precise. I have replicated prospective evidence that a tiny,
selected set of SVD atoms causally participates in this post-training
regression and beats arbitrary same-budget sparse edits. I do not yet claim
that this selector always beats robust FoBa or energy, or that it generalizes
to every behavior. The next test is a second frozen regression and another
model family.
