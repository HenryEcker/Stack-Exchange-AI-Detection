from datetime import datetime, timedelta
from typing import Iterable, Generator, TypedDict

import pandas as pd
from html2text import html2text
from stackapi import StackAPI
from transformers import pipeline, RobertaTokenizer, Pipeline

LABEL_DICT = {"LABEL_0": "AI Generated", "LABEL_1": "Human"}
ROBERTA_CONFIG = {
    'model': 'roberta-base-openai-detector',
    'truncation': True
}
CLASS_PIPELINE = pipeline(model=ROBERTA_CONFIG['model'], truncation=ROBERTA_CONFIG['truncation'])
TOKENIZER = RobertaTokenizer.from_pretrained(ROBERTA_CONFIG['model'], truncation=ROBERTA_CONFIG['truncation'])
SE_API = StackAPI('stackoverflow', key='wuBIf6UmmChWWUNCAM*Rmg((')


class NoResultsException(Exception):
    def __init__(self, *args):
        super(NoResultsException, self).__init__(self, *args)


class Post(TypedDict):
    body: str
    post_id: str
    title: str


class PostResults(TypedDict):
    ID: str
    Title: str
    Length: int
    Class: str
    Score: float


def process_items(items: Iterable[Post]) -> Generator[PostResults, None, None]:
    for item in items:
        body = html2text(item['body'])
        result = CLASS_PIPELINE(body)
        yield {
            'ID': item['post_id'],
            'Title': item['title'],
            'Length': len(TOKENIZER(body)['input_ids']),
            'Class': LABEL_DICT[result[0]['label']],
            'Score': round(result[0]['score'], 3)
        }


def lookup_user(user_id: int, time_delta=timedelta(days=2)) -> pd.DataFrame:
    from_date = datetime.now() - time_delta

    answers = SE_API.fetch(
        'users/{ids}/answers',
        fromdate=from_date,
        ids=[user_id]
    )

    answer_ids = [item['answer_id'] for item in answers['items']]
    if not answer_ids:
        raise NoResultsException('This user has no answers!')

    answers_text = SE_API.fetch(
        'posts',
        ids=answer_ids,
        filter='!LH22Vfx-WtNBnMCP-eADaa'
    )

    return pd.DataFrame(process_items(answers_text['items']))


def lookup_post(post_id: int) -> Pipeline:
    post = SE_API.fetch('posts', ids=[post_id], filter='!LH22Vfx-WtNBnMCP-eADaa')
    items = post['items']
    if not items:
        raise NoResultsException('No content found for this post!')
    body = html2text(post['items'][0]['body'])
    return CLASS_PIPELINE(body)


def main():
    for post_id in [74680771]:
        try:
            result = lookup_post(post_id)
            print(f"Result: {LABEL_DICT[result[0]['label']]} - Score: {str(round(result[0]['score'], 3))}")
        except NoResultsException:
            pass  # Or something

    for user_id in [20704445]:
        try:
            result_df = lookup_user(user_id)
            with pd.option_context('display.width', None):
                print(result_df)
        except NoResultsException:
            pass  # Or something


if __name__ == '__main__':
    main()
