# AutoMind Core Module

## Note

### Pyright: extra typings

```py
pyright --createstub featuretools
pyright --createstub imblearn
```

### Add New DC/FE Method

1. add new definition to model → preprocessing.py → class
   DataCleaningRecommendations/FeatureEngineeringRecommendations

1. add new key attribute to data_utils → template.py → function
   get_few_shot_prompt → modeling_approaches → data_cleaning/feature_engineering

1. add defined methods to logic_applier.py -> apply_llm_recommendations
