# Prospective causal calibration v1 stop record

The v1 prospective run stopped before confirmation.

The first completed selection evaluation, seed 727, showed that the shared base model scored only 4/12 on both `quoted_a` and `quoted_b`, and only 10/12 on the marker target. The frozen input validity gate required all 12 marker targets and at least 11/12 in every protected family. Because the base model is identical across all five seeds, this shared precondition failure deterministically rejects the full campaign before atom selection.

No confirmation evaluation ran. The failure is retained as a data-design error: the source builder used the position-bias capability screen, which did not screen the quoted-instruction controls later required by the protocol.

The next protocol must use fresh sources and seeds, explicitly align its protected families with the base capability screen, and execute the shared input gate before computing gradients or candidate supports.
