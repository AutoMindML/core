from enum import Enum
from typing import Type, cast

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

method_registry = {}


# this solution is from pydantic issue: https://github.com/pydantic/pydantic/discussions/2980#discussioncomment-12977507
class EnumByName:
    """Pydantic validator for Enum fields that accepts both enum instances and string names."""

    def __init__(self, *, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def __get_pydantic_core_schema__(
        self, enum_cls: Type[Enum], _handler: GetCoreSchemaHandler
    ):
        name_enum = Enum(
            "name_enum", {member.name: member.name for member in enum_cls}
        )
        name_enum = cast(type[Enum], name_enum)

        def enum_or_name(value: Enum | str) -> Enum:
            if isinstance(value, str):
                if not self.ignore_case:
                    try:
                        return enum_cls[value]
                    except KeyError:
                        raise ValueError(f"Enum name not found: {value}")
                try:
                    return next(
                        member
                        for member in enum_cls
                        if member.name.lower() == value.lower()
                    )
                except StopIteration:
                    raise ValueError(f"Enum name not found: {value}")
            elif isinstance(value, enum_cls):
                return value
            raise ValueError(
                f"Expected enum member or name, got {type(value).__name__}: {value}"
            )

        return core_schema.no_info_plain_validator_function(
            enum_or_name,
            json_schema_input_schema=core_schema.enum_schema(
                enum_cls, list(name_enum.__members__.values())
            ),
            ref=enum_cls.__name__,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda e: e.name
            ),
        )


def register_method(method: Enum):
    """Decorator to register methods in the method registry."""

    def decorator(func):
        method_registry[method.name] = func
        return func

    return decorator
