import pandas as pd
from transformers import pipeline, RobertaTokenizer
from stackapi import StackAPI
from html2text import html2text
from datetime import datetime, timedelta


class_pipeline = pipeline(model='roberta-base-openai-detector',
                          truncation=True)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base-openai-detector',truncation=True)

label_dict = {"LABEL_0" : "AI Generated", "LABEL_1" : "Human"}

api = StackAPI('stackoverflow',key='wuBIf6UmmChWWUNCAM*Rmg((')
for id in [74680771]:
    post = api.fetch('posts', ids=[id], filter='!LH22Vfx-WtNBnMCP-eADaa')
    body = html2text(post['items'][0]['body'])

    result = class_pipeline(body)
    print('Result: '+label_dict[result[0]['label']]+' - Score: '+str(round(result[0]['score'],3)))

for userid in [20704445]:
    lookback = datetime.now() - timedelta(days=2)
    answers = api.fetch('users/{ids}/answers', fromdate=lookback, todate=datetime.now(), ids=[userid])
    answer_ids = []
    for item in answers['items']:
        answer_ids.append(item['answer_id'])

    answers_text = api.fetch('posts', ids = answer_ids, filter='!LH22Vfx-WtNBnMCP-eADaa')
    id_list = []
    title_list = []
    length_list = []
    class_list = []
    score_list = []
    for text in answers_text['items']:
        body = html2text(text['body'])
        result = class_pipeline(body)
        id_list.append(text['post_id'])
        title_list.append(text['title'])
        length_list.append(len(tokenizer(body)['input_ids']))
        class_list.append(label_dict[result[0]['label']])
        score_list.append(round(result[0]['score'],3))
    result_df = pd.DataFrame({'ID':id_list,
                              'Title':title_list,
                              'Length':length_list,
                              'Class':class_list,
                              'Score':score_list})
    print(result_df)

