from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition as fr
from datetime import datetime

import json
import base64

app = Flask(__name__)
CORS(app)  # Habilita o CORS

# Carregar rostos conhecidos
rostos_conhecidos = []
nomes_dos_rostos = []

matheus1 = fr.face_encodings(fr.load_image_file("./img/matheus1.jpg"))
if matheus1:
    rostos_conhecidos.append(matheus1[0])
    nomes_dos_rostos.append("Matheus Cardoso")

video_capture = None  # Usado para armazenar o objeto de captura de vídeo

@app.route('/open_webcam', methods=['POST'])
def open_webcam():
    global video_capture
    if video_capture is not None:
        return jsonify({'mensagem': 'A webcam ja esta aberta'})

    video_capture = cv2.VideoCapture(0)
    return jsonify({'mensagem': 'Webcam aberta'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        return jsonify({'mensagem': 'Reconhecimento parado'})
    else:
        return jsonify({'mensagem': 'Nenhum reconhecimento em andamento'})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    global video_capture
    if video_capture is None:
        return jsonify({'erro': 'A webcam nao esta aberta'})

    try:
        content = request.get_json(silent=True)
        if not content or 'imagem' not in content:
            return jsonify({'erro': 'Nenhuma imagem recebida'})

        # Decodificar a imagem base64
        imagem_base64 = content['imagem']
        imagem_decodificada = base64.b64decode(imagem_base64.split(',')[1])

        # Converter a imagem para o formato numpy
        imagem_np = np.frombuffer(imagem_decodificada, dtype=np.uint8)
        frame = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR)

        # Salvar a imagem capturada no sistema de arquivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imagem_salva_path = f'./img/imagem_capturada_{timestamp}.jpg'
        cv2.imwrite(imagem_salva_path, frame)
        print(f'Imagem capturada salva como {imagem_salva_path}')

        # Converta a imagem para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Localize rostos na imagem
        localizacao_dos_rostos = fr.face_locations(rgb_frame)

        if not localizacao_dos_rostos:
            return jsonify({'erro': 'Nenhum rosto detectado na imagem'})

        rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

        nomes_detectados = []
        for rosto_desconhecido in rosto_desconhecidos:
            resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)

            if resultados:
                melhor_id = np.argmax(resultados)

                if melhor_id is not None and melhor_id < len(nomes_dos_rostos):
                    nome_detectado = nomes_dos_rostos[melhor_id]
                else:
                    nome_detectado = "Desconhecido"
            else:
                nome_detectado = "Desconhecido"

            nomes_detectados.append(nome_detectado)

        # Retornar a resposta como JSON
        response_data = {'nomes_detectados': nomes_detectados, 'mensagem': 'Reconhecimento realizado com sucesso'}
        print(response_data)
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'erro': f'Erro ao processar a imagem'})


if __name__ == '__main__':
    app.run(debug=True)
