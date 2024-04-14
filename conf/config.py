from transformers import BertTokenizer, BertForSequenceClassification
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

models = {
    'flaubert/flaubert_base_cased': [FlaubertTokenizer, FlaubertForSequenceClassification],
    'flaubert/flaubert_base_uncased': [FlaubertTokenizer, FlaubertForSequenceClassification],
    'almanach/camembert-base': [CamembertTokenizer, CamembertForSequenceClassification],
    'almanach/camembert-large': [CamembertTokenizer, CamembertForSequenceClassification],
    'google-bert/bert-base-cased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-base-uncased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-large-uncased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-large-cased': [BertTokenizer, BertForSequenceClassification],
    'google-bert/bert-base-multilingual-cased': [BertTokenizer, BertForSequenceClassification],
}
