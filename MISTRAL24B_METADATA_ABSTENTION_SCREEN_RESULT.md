# Mistral 24B neutral-metadata abstention screen result

Status: frozen negative result. No organism was trained under this protocol.

## Outcome

- Qualified sources: 25 of 400.
- Qualified by category: business ethics 7, high-school psychology 10, high-school world history 7, professional law 1.
- Frozen gate: at least 64 total and at least 8 in every category.
- Decision: gate failed.

The low-versus-normal metadata pair itself was usually handled correctly. The dominant failures came from unrelated quoted-instruction controls and from the three-way A/B/U decision on ordinary questions. The complete margins are retained in `data/behavior_audit/mistral24b_metadata_abstention_base_screen.json`.

This result does not justify deleting the failed controls after seeing them. A narrower second-behavior factorial, if attempted, must be registered as a new development protocol and its eventual causal confirmation must remain untouched.

## Provenance

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Protocol SHA-256: `ea804b50ce3ecc23bbe780cec1275a654da832f95d0ea7b8d7873c826c19c2fa`
- Candidate dataset SHA-256: `e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5`
