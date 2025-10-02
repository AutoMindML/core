from typing import List, Optional, TypedDict

from featuretools.entityset.entityset import EntitySet
from featuretools.entityset.relationship import Relationship
from featuretools.synthesis.dfs import dfs

# from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
# from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from pandas import DataFrame

from automind.utils.logging import logger


class PrimitiveDict(TypedDict):
    agg: List[str]
    transform: List[str]


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

        self._primitives: PrimitiveDict = {"agg": [], "transform": []}

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
                )
                logger.info("apply dfs successfully!")
            except Exception:
                logger.error(
                    "error when apply dfs to entity set with target entity"
                )

            return

        logger.error("target entity must specific")
