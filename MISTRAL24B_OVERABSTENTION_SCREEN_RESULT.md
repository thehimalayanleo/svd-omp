# Mistral 24B warning-triggered over-abstention screen result

Status: frozen negative result. The screen was run after the protocol hash was fixed and before any organism training.

## Outcome

- Qualified sources: 42 of 400.
- Qualified by category: business ethics 10, high-school psychology 16, high-school world history 13, professional law 3.
- Frozen promotion gate: at least 80 total and at least 16 in every category.
- Decision: gate failed. No organism may be trained under this protocol.

The model often treated the explicit quality warning as meaningful uncertainty, especially for professional-law items. That makes the proposed behavior scientifically confounded: post-training would amplify a behavior already elicited by the warning rather than introduce a clean metadata-triggered regression.

The complete per-source margins are retained in `data/behavior_audit/mistral24b_overabstention_base_screen.json`. A revised experiment must use a new protocol and a semantically neutral metadata field. It must not retroactively change this gate or exclude the failed category.

## Provenance

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Protocol SHA-256: `0e9df02a1f158c74fafb9e96573abaf4b37ddb437d11ecae46b4debd695f7fb1`
- Candidate dataset SHA-256: `e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5`
