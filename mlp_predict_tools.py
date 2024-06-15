import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import subprocess
import csv
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neural_network import MLPClassifier

def imagem_para_lista_de_coordenadas(caminho_imagem, tecnica, hands):
    try:
        image = cv2.imread(caminho_imagem)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Processando a imagem com o MediaPipe Hands
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            newRow = []
            if(tecnica == "centro_geometrico"):
                results = results.multi_hand_world_landmarks
            else:
                results = results.multi_hand_landmarks
            for hand_landmarks in results:
                for landmark in hand_landmarks.landmark:
                    newRow.append(landmark.x)
                    newRow.append(landmark.y)
                    newRow.append(landmark.z)
            return newRow
        else:
            return None
    except Exception as e:
        return None

def contar_arquivos(caminho_repositorio):
    count_subprocess_result = subprocess.run(f'find {caminho_repositorio} -type f | wc -l', stdout=subprocess.PIPE, shell=True, text=True)
    output = count_subprocess_result.stdout.strip()
    return int(output)

# tipo = <train | test>
# tecnica= <centro_geometrico | normalizado>
# ATENÇÃO: o código assume a existência de diretorios destinados as imagens 
# as quais o mediapipe não foi capaz de extrair as landmarks
# nomeados como imagens_com_problemas_{tipo}_{tecnica}

def gerador_de_csv_com_as_coordenadas_e_os_rotulos(tipo, tecnica):
    # Obtendo o numero de arquivos de imagem de cada letra
    # Lista com nome de todos os diretórios
    label = ["A","B","C","D","E","F","G","I","L","M","N","O","P","Q","R","S","T","U","V","W"]
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    num_files = []# cada elemento da lista é o numero de arquivos de imagem das letras em ordem alfabética
    for i,reps in enumerate(label):
        # Caminho do diretório
        directory = f'./imagens/{tipo}/{reps}'
        num_files.append(contar_arquivos(directory))
    
    csv_filename = f'{tipo}_{tecnica}.csv'
    header = [f'Landmark{i}_{axis}' for i in range(0, 21) for axis in ['x', 'y', 'z']]
    header.append('label')
    
    command = f'rm -rf ./imagens_com_problemas_{tipo}_{tecnica}/*',
    subprocess.run(command, shell=True, check=True)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header) # Inserindo os nomes dos atributos
        for i, reps in enumerate(label): 
            for j in range(1,num_files[i]+1):

                # Outro problema com os nomes
                if(reps == 'F' and j > 300 and tipo == 'test'):
                    k = j + 100
                else:
                    k = j

                caminho = f"imagens/{tipo}/{reps}/{k}.png"

                lista_de_coordenadas = imagem_para_lista_de_coordenadas( caminho, tecnica, hands)
                if lista_de_coordenadas is not None:
                    lista_de_coordenadas.append(reps)
                    writer.writerow(lista_de_coordenadas)
                else:
                    try:
                        image = cv2.imread(caminho)
                        cv2.imwrite(f"imagens_com_problemas_{tipo}_{tecnica}/imagem{reps}00{k}.png", image)
                    except Exception as e:
                        print("Caminho com problema")
                        print(caminho)


