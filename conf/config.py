from transformers import BertTokenizer, BertForSequenceClassification
from transformers import CamembertTokenizer, CamembertForSequenceClassification

models = {
    'almanach/camembert-base': [CamembertTokenizer, CamembertForSequenceClassification],
    'almanach/camembert-large': [CamembertTokenizer, CamembertForSequenceClassification],
    'google-bert/bert-base-cased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-base-uncased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-large-uncased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-large-cased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-base-multilingual-cased': [BertTokenizer, BertForSequenceClassification],
}
