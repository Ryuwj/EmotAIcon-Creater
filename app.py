from flask import Flask, request, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일의 내용을 수동으로 로드
with open('environment.env', 'r') as file:
    for line in file:
        if 'OPENAI_API_KEY' in line:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

# 이제 환경 변수 사용
api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        translated_input = translate_to_english(user_input)
        return render_template('index.html', translated=translated_input)
    return render_template('index.html', translated='')

def translate_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": ("This assistant will translate Korean sentences into English for creating emoji images with AI art generators.\n"
                                "For each Korean sentence, provide the English translation. For example, translate descriptions of characters, scenes, or actions.")
                },
                # few-shot 학습 예제
                {
                    "role": "user",
                    "content": "명일방주의 아미야가 옥상에서 손 흔들고 있는 이미지"
                },
                {
                    "role": "assistant",
                    "content": "An image of Amiya from Arknights waving her hand on the rooftop."
                },
                # 추가적인 few-shot 학습 예제들 (필요한 경우 여기에 추가)
                # 사용자 입력
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=1,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
