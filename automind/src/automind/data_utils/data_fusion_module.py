from typing import TypedDict

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset.entityset import EntitySet
from featuretools.entityset.relationship import Relationship
from featuretools.synthesis.dfs import dfs
from pandas import DataFrame

from automind.models.primitive import (
    AggregationPrimitive,
    DefaultAggregationPrimitive,
    DefaultTransformPrimitive,
    TransformPrimitive,
)
from automind.utils.logging import logger


class PrimitiveDict(TypedDict):
    agg: list[AggregationPrimitive]
    transform: list[TransformPrimitive]


class DataFusionModule:
    def __init__(
        self,
        dfm_id: str | None = None,
        target_entity_name: str | None = None,
    ) -> None:
        if (target_entity_name is not None) and (dfm_id is None):
            dfm_id = target_entity_name

        if dfm_id:
            self.entity_set = EntitySet(id=dfm_id)
        else:
            self.entity_set = EntitySet()

        self.target_entity_name: str | None = target_entity_name
        self.feature_matrix: DataFrame | None = None

        self._relationships: list[Relationship] = []
        self._feature_defs: list | None = None

        self._primitives: PrimitiveDict = {
            "agg": DefaultAggregationPrimitive,
            "transform": DefaultTransformPrimitive,
        }

    def set_target(self, target_entity_name: str):
        self.target_entity_name = target_entity_name

    def add_entity(self, data: DataFrame, entity_name: str, entity_pk: str):
        self.entity_set.add_dataframe(data, entity_name, entity_pk)

    def add_relationship(
        self,
        parent_entity_name: str,
        parent_entity_pk: str,
        child_entity_name: str,
        child_entity_fk: str,
    ):
        relationship = Relationship(
            self.entity_set,
            parent_entity_name,
            parent_entity_pk,
            child_entity_name,
            child_entity_fk,
        )

        self._relationships.append(relationship)
        self.entity_set.add_relationship(relationship=relationship)

    def apply_dfs(self) -> DataFrame:
        if not self.target_entity_name:
            raise ValueError("target entity must be specified")

        try:
            self.feature_matrix, self._feature_defs = dfs(
                entityset=self.entity_set,
                target_dataframe_name=self.target_entity_name,
                agg_primitives=self._primitives["agg"],
                trans_primitives=self._primitives["transform"],
            )
        except Exception as error:
            raise ValueError(
                f"failed to apply DFS for target {self.target_entity_name!r}"
            ) from error

        logger.info("apply dfs successfully!")
        return self.feature_matrix

    def get_deep_feature_dataframe(self):
        return self.feature_matrix

    def get_feature_definitions(self) -> list:
        if self._feature_defs is None:
            raise ValueError(
                "feature definitions are unavailable; call apply_dfs first"
            )
        return self._feature_defs.copy()

    def apply_feature_definitions(self, feature_definitions: list) -> DataFrame:
        if not feature_definitions:
            raise ValueError("feature definitions must not be empty")
        try:
            self.feature_matrix = calculate_feature_matrix(
                features=feature_definitions,
                entityset=self.entity_set,
            )
        except Exception as error:
            raise ValueError(
                "failed to calculate frozen feature definitions"
            ) from error
        return self.feature_matrix

    def set_primitives(
        self,
        agg: list[AggregationPrimitive],
        transform: list[TransformPrimitive],
    ):
        self._primitives.update({"transform": transform, "agg": agg})

    def plot_entity_set(self, path: str):
        self.entity_set.plot(path)

    def get_entity_set_relationships(self):
        return self.entity_set.to_dictionary()

    @staticmethod
    def get_default_primitives() -> PrimitiveDict:
        return {
            "transform": DefaultTransformPrimitive,
            "agg": DefaultAggregationPrimitive,
        }
