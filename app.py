from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
from openai import OpenAI
import requests
import os
import base64
import json
import io
import hashlib
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime


# .env 파일의 내용을 수동으로 로드
with open('environment.env', 'r') as file:
    for line in file:
        if 'OPENAI_API_KEY' in line:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

# 이제 환경 변수 사용
api_key = os.getenv('OPENAI_API_KEY')

print("api_key : " + api_key)



app = Flask(__name__)
load_dotenv()
client = OpenAI(api_key=api_key)


STABLE_DIFFUSION_API_URL = "http://localhost:7860"  # 실제 로컬 API 엔드포인트로 교체하세요.

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        translated_prompt = translate_to_english(user_input)
        # image_data = generate_image(translated_prompt)
        
        # 현재 시간을 문자열로 포맷팅
        
        # 이미지가 저장된 경로 지정
        image_path = generate_image(translated_prompt)
        
        print("translated_prompt : " + translated_prompt)
        
        # image_b64 = base64.b64encode(image_data).decode('utf-8')
        return render_template('index.html', image_path=image_path, prompt=translated_prompt)
    return render_template('index.html')

def translate_to_english(text):
    try:
        
        print("user input : " + text)
        
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
        # 챗 API의 최신 버전에서는 'message' 키를 이렇게 접근합니다.
        answer = response.choices[0].message.content
        
        print("OpenAI API answer : " + answer)
        
        return answer
    except Exception as e:
        return str(e)


def generate_image(prompt):
    
    # Stable Diffusion 에 프롬프트로 넣을 값
    # payload 에는 프롬프트와 옵션을 같이 넣음
    payload = {
        "prompt": prompt,
        "steps": 20
    }
    
    response = requests.post(url=f'{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img', json=payload)
    
    response_json = response.json()
        
    image_data = Image.open(io.BytesIO(base64.b64decode(response_json['images'][0])))
    
    # 현재 날짜와 시간으로 파일 이름 해시 생성
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    hashed_name = hashlib.sha256(current_time.encode()).hexdigest()
    filename = f"{hashed_name}.png"
    
    folder_path = create_folder_for_today('static/images')
    
    # 이미지 파일 저장
    image_path = os.path.join(folder_path, filename)
    image_data.save(image_path)
    
    return image_path

@app.route('/regenerate', methods=['POST'])
def regenerate():
    prompt = request.form['prompt']
    #image_data = generate_image(prompt)
    #image_b64 = base64.b64encode(image_data).decode('utf-8')
    image_path = generate_image(prompt)
    
    return render_template('index.html', image_path=image_path, prompt=prompt)

@app.route('/download', methods=['POST'])
def download():
    image_path = request.form['image_path']
    
    directory = os.path.join(app.root_path, image_path)
    
    return send_file(directory, as_attachment=True)

def create_folder_for_today(base_dir):
    # 오늘 날짜로 폴더 이름 생성
    today = datetime.now().strftime("%Y%m%d")
    folder_path = os.path.join(base_dir, today)

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

if __name__ == '__main__':
    app.run(debug=True)
