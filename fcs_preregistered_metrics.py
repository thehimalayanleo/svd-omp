"""Pure source-paired metrics for the preregistered FCS validation."""


def factorial_specificity(
    repaired_ids: list[str], paired_correct_ids: list[str], baseline_paired_ids: list[str]
) -> dict:
    repaired = {item.removeprefix("benign_marker:") for item in repaired_ids}
    paired_preserved = {item.removeprefix("marked_ambiguous:") for item in paired_correct_ids}
    baseline_paired = {item.removeprefix("marked_ambiguous:") for item in baseline_paired_ids}
    specific = repaired & paired_preserved
    shortcut = repaired - paired_preserved
    damage = baseline_paired - paired_preserved
    return {
        "gross_repairs": len(repaired),
        "specific_repairs": len(specific),
        "shortcut_repairs": len(shortcut),
        "paired_damage": len(damage),
        "net_specific_repair": (len(specific) - len(damage)) / 24,
        "specific_source_ids": sorted(specific),
        "shortcut_source_ids": sorted(shortcut),
        "damaged_source_ids": sorted(damage),
    }
