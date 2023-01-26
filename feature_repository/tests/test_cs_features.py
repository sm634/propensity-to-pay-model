from feature_repository.features.cos_wday import CosWeekDay
from preprocessing_layer.engineer_features import get_data

data = get_data(query_data=False)
feature = CosWeekDay(data)
feature.logic()

print(feature.feature)
