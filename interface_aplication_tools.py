import cv2
import numpy as np

# Função para exibir o resultado da classificação na câmera
def display_classification(frame, predicted_letter):
    # Adicionar o texto da classificação na imagem
    cv2.putText(frame, f'Classificacao: {predicted_letter}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Exibir a imagem com a classificação
    cv2.imshow('Classificacao de Libras', frame)

# Função para pré-processamento da imagem
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Função para exibir o resultado da classificação
def display_prediction(img):
    cv2.imshow('Camera', img)