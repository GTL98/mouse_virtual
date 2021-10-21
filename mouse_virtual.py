################################################################################################
#                                                                                              #
#  Toda a lógica mais detalhada está presente no arquivo "Contador de Dedos.ipynb"             #
#                                                                                              #
#  Em caso de dúvidas, consultar a documentação:                                               #
#      - "Aula 1 - Rastreamento de mão (Introdução).ipynb" no link abaixo.                     #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


# Importar as bibliotecas
import cv2
import time
import autopy
import numpy as np
import rastreamento_mao as rm

# Definir o tamanho da tela
largura_tela = 640
altura_tela = 480

# Taxa de frame (FPS)
tempo_anterior = 0
tempo_atual = 0

# Módulo DetectorMao
detector = rm.DetectorMao(max_maos=1, deteccao_confianca=0.7, rastreamento_confianca=0.7)

# Tamanho da tela do computador
largura_computador, altura_computador = autopy.screen.size()

# Delimitar o espaço que o programa reconhece o dedo
limite_captura = 100

# Suavidade do mouse
suavidade = 5

# Captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(3, largura_tela)
cap.set(4, altura_tela)

x_anterior, y_anterior = 0, 0
x_atual, y_atual = 0, 0

while True:
    # 1. Encontrar as landmarks da mão
    sucesso, imagem = cap.read()
    imagem = detector.encontrar_maos(imagem)
    lista_landmark, caixa_limite = detector.encontrar_posicao(imagem)
    
    # 2. Captar a pontas dos dedos indicador e médio
    if lista_landmark:
        # Ponta do indicador
        x1, y1 = lista_landmark[8][1:]
        
        # Ponta do médio
        x2, y2 = lista_landmark[12][1:]
        
        # 3. Checar os dedos levantados
        dedos = detector.dedos_levantados()
        cv2.rectangle(imagem, (limite_captura, limite_captura),
                      (largura_tela-limite_captura, altura_tela-limite_captura), (0, 0, 255), 2)
    
        # 4. Somente o indicador: mover o mouse
        if dedos[1] == 1 and dedos[2] == 0:
            # 5. Converter as cordenadas
            x_mouse = np.interp(x1, (limite_captura, largura_tela-limite_captura), (0, largura_computador))
            y_mouse = np.interp(y1, (limite_captura, altura_tela-limite_captura), (0, altura_computador))

            # 6. Suavidade do movimento
            x_atual = x_anterior + (x_mouse - x_anterior) / suavidade
            y_atual = y_atual + (y_mouse - y_anterior) / suavidade

            # 7. Movimento do mouse
            autopy.mouse.move(largura_computador-x_atual, y_atual)
            cv2.circle(imagem, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            x_anterior, y_anterior = x_atual, y_atual

        # 8. Indicador e médio levantados: clicar
        if dedos[1] == 1 and dedos[2] == 1:
            # 9. Encontrar a distância entre os dedos
            comprimento, imagem, info_linha = detector.encontrar_distancia(8, 12, imagem)
            # 10. Clicar se a distância entre os dedos for pequena
            if comprimento < 30:
                cv2.circle(imagem, (info_linha[4], info_linha[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    
    # 11. Mostrar o FPS
    tempo_atual = time.time()
    fps = 1/(tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual
    cv2.putText(imagem, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # 12. Mostrar a imagem na tela
    cv2.imshow('Mouse Virtual', imagem)
    
    # 13. Terminar o loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
        
# 14. Fechar a tela de captura
cap.release()
cv2.destroyAllWindows()
