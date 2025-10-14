from typing import List, Optional, TypedDict

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

__all__ = ["DataFusionModule"]


class PrimitiveDict(TypedDict):
    agg: List[AggregationPrimitive]
    transform: List[TransformPrimitive]


class DataFusionModule:
    def __init__(
        self,
        dfm_id: Optional[str] = None,
        target_entity_name: Optional[str] = None,
    ) -> None:
        if (target_entity_name is not None) and (dfm_id is None):
            dfm_id = target_entity_name

        if dfm_id:
            self.entity_set = EntitySet(id=dfm_id)
        else:
            self.entity_set = EntitySet()

        self.target_entity_name: Optional[str] = target_entity_name
        self.feature_matrix: Optional[DataFrame] = None

        self._relationships: List[Relationship] = []
        self._feature_defs: Optional[List] = None

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

    def apply_dfs(self):
        if self.target_entity_name:
            try:
                self.feature_matrix, self._feature_defs = dfs(
                    entityset=self.entity_set,
                    target_dataframe_name=self.target_entity_name,
                    agg_primitives=self._primitives["agg"],
                    trans_primitives=self._primitives["transform"],
                )
                logger.info("apply dfs successfully!")
            except Exception:
                logger.error(
                    "error when apply dfs to entity set with target entity"
                )

            return

        logger.error("target entity must specific")

    def get_deep_feature_dataframe(self):
        return self.feature_matrix

    def set_primitives(
        self,
        agg: List[AggregationPrimitive],
        transform: List[TransformPrimitive],
    ):
        self._primitives.update({"transform": transform, "agg": agg})

    @staticmethod
    def get_default_primitives() -> PrimitiveDict:
        return {
            "transform": DefaultTransformPrimitive,
            "agg": DefaultAggregationPrimitive,
        }
