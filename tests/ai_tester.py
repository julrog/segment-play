import base64
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
API_ORG = os.getenv('OPENAI_ORG_ID')


class AITester:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=API_KEY, organization=API_ORG)

    def encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def binary_question(self, image: np.ndarray, question: str) -> str:
        prompt = f'{question} Answer the question with just a single word "True" or "False".'  # noqa E501
        image_url = f'data:image/jpeg;base64,{self.encode_image(image)}'
        response = self.client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': image_url,
                                'detail': 'low'
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content == 'True'
