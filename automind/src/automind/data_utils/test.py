from automind.data_utils.preprocessing import LLMOutputSchema

parsed_json = "<json>...</json>"
validated_json = LLMOutputSchema.model_validate(parsed_json)

validated_json.modeling_approaches[0].data_cleaning.missing_values
