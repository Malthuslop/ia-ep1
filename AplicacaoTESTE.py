import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mlp_predict_tools import imagem_para_lista_de_coordenadas

# Carregar o modelo treinado
model = load_model('modelo_libras.h5')  # Substitua pelo caminho do seu modelo

# Função para exibir o resultado da classificação na câmera
def display_classification(frame, predicted_letter):
    # Adicionar o texto da classificação na imagem
    cv2.putText(frame, f'Classificação: {predicted_letter}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Exibir a imagem com a classificação
    cv2.imshow('Classificação de Libras', frame)

# Função para pré-processamento da imagem
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0
    return img

# Função para exibir o resultado da classificação
def display_prediction(img, label):
    cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Camera', img)

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame da câmera")
        break
    
    # Mostrar a câmera com um label inicial
    display_prediction(frame, 'Pressione "Espaco" para classificar e "Q" para sair')

    # Verificar se a tecla "Espaço" foi pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        # Tirar uma foto (frame) e pré-processar
        photo = preprocess_image(frame)
        
        # Fazer a previsão usando o modelo
        prediction = model.predict(photo)
        predicted_class = np.argmax(prediction)
        
        # Mapear a classe prevista para a letra correspondente
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
        predicted_letter = classes[predicted_class]

        # Exibir o resultado da classificação na câmera usando o OpenCV
        display_classification(frame, predicted_letter)

    # Fechar a câmera ao pressionar a tecla "q"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
