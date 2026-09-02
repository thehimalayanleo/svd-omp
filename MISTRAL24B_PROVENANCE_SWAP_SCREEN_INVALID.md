# Invalid provenance-header screen implementation

The first Modal screen completed, but its target row encoded `positive_completion=B` and `negative_completion=A`. The shared untouched-base screener correctly evaluates a target against `negative_completion`, so the implementation asked the base model to answer A. This contradicted the frozen protocol, which says the untouched model must answer B and a future organism would learn A.

The result (`1/400` qualified) is therefore not evidence about the preregistered screen. It is retained as an invalid implementation run. The only correction swaps the target row to `positive_completion=A`, `negative_completion=B`, matching the already frozen prose. The protocol, model, prompt, margin gate, promotion gate, and all other families remain unchanged.
