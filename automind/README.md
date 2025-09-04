# AutoMind Core Module

## Feature

- data preprocessing pipeline

## Note

### How to add new DC or FE method?

1. add new definition to model → preprocessing.py → class
   DataCleaningRecommendations/FeatureEngineeringRecommendations

1. add new key attribute to data_utils → template.py → function
   get_few_shot_prompt → modeling_approaches → data_cleaning/feature_engineering
