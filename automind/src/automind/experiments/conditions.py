from dataclasses import dataclass

from automind.experiments.protocol import Condition


@dataclass(frozen=True)
class ConditionSpec:
    use_dfm: bool
    use_llm: bool
    use_tpot: bool
    use_sampling: bool
    use_feature_engineering: bool


CONDITIONS = {
    Condition.C0_DETERMINISTIC: ConditionSpec(
        True, False, False, False, False
    ),
    Condition.C1_TPOT: ConditionSpec(True, False, True, False, False),
    Condition.C2_AUTOMIND_FIXED: ConditionSpec(True, True, False, True, True),
    Condition.C3_AUTOMIND_TPOT: ConditionSpec(True, True, True, True, True),
    Condition.C4_WITHOUT_DFM: ConditionSpec(False, True, True, True, True),
    Condition.C5_WITHOUT_LLM: ConditionSpec(True, False, True, True, True),
    Condition.C6_WITHOUT_SAMPLING: ConditionSpec(True, True, True, False, True),
    Condition.C7_WITHOUT_FE: ConditionSpec(True, True, True, True, False),
}


def condition_spec(condition: Condition, *, relational: bool) -> ConditionSpec:
    if condition == Condition.C4_WITHOUT_DFM and not relational:
        raise ValueError("C4 is not applicable to a flat dataset")
    return CONDITIONS[condition]
