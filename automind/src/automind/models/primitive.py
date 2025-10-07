from typing import List, Literal

AggregationPrimitive = Literal[
    "sum",
    "std",
    "max",
    "skew",
    "min",
    "mean",
    "count",
    "percent_true",
    "num_unique",
    "mode",
]

DefaultAggregationPrimitive: List[AggregationPrimitive] = [
    "sum",
    "std",
    "max",
    "skew",
    "min",
    "mean",
    "count",
    "percent_true",
    "num_unique",
    "mode",
]


TransformPrimitive = Literal[
    "day",
    "year",
    "month",
    "weekday",
    "haversine",
    "num_words",
    "num_characters",
]

DefaultTransformPrimitive: List[TransformPrimitive] = [
    "day",
    "year",
    "month",
    "weekday",
    "haversine",
    "num_words",
    "num_characters",
]
