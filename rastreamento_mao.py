################################################################################################
#                                                                                              #
#  Todo esse código será feito na extensão .py para que seja possível usá-lo nos projetos.     #
#                                                                                              #
#  Todas as anotações e explicações sobre o que está sendo usado nesse documento podem ser     #
#  encontradas no documento "Aula 1 - Rastramento de mão (Introdução).ipynb".                  #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################


import cv2
import mediapipe as mp
import time
import math


class DetectorMao:
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5, rastreamento_confianca=0.5):
        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreamento_confianca = rastreamento_confianca


        # Mãos
        self.mpMaos = mp.solutions.hands
        self.maos = self.mpMaos.Hands(self.modo, self.max_maos, self.deteccao_confianca, self.rastreamento_confianca)

        # Desenhar as landmarks
        self.mp_desenho = mp.solutions.drawing_utils
        
        # Landmarks das pontas dos dedos
        self.landmarks_ponta_dedos = [4, 8 , 12, 16, 20]

    def encontrar_maos(self, imagem, desenho=True):
        # Converter a cor da imagem (o Mediapipe usa somente imagens em RGB e o OpenCV captura em BGR)
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Resultado do processamento da imagem
        self.resultados = self.maos.process(imagem_rgb)

        # Colocar as landmarks na mão
        if self.resultados.multi_hand_landmarks:
            for mao_landmark in self.resultados.multi_hand_landmarks:
                if desenho:
                    self.mp_desenho.draw_landmarks(imagem, mao_landmark, self.mpMaos.HAND_CONNECTIONS,
                                             self.mp_desenho.DrawingSpec(color=(0, 0, 255)),
                                             self.mp_desenho.DrawingSpec(color=(0, 255, 0)))
                    
        return imagem
    
    def encontrar_posicao(self, imagem, num_mao=0, desenho=True):
        lista_x = []
        lista_y = []
        caixa_limite = []
        self.lista_landmark = []
        
        if self.resultados.multi_hand_landmarks:
            minha_mao = self.resultados.multi_hand_landmarks[num_mao]
            
            for item, landmark in enumerate(minha_mao.landmark):
                altura, largura, centro = imagem.shape
                cx, cy = int(landmark.x*largura), int(landmark.y*altura)
                lista_x.append(cx)
                lista_y.append(cy)
                self.lista_landmark.append([item, cx, cy])
                
            x_min, x_max = min(lista_x), max(lista_x)
            y_min, y_max = min(lista_y), max(lista_y)
            caixa_limite = x_min, y_min, x_max, y_max
                
            if desenho:
                    cv2.rectangle(imagem, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
            
        return self.lista_landmark, caixa_limite
    
    def dedos_levantados(self):
        dedos = []
        # Loop para o dedão (usa o eixo X para verificar se está levantando ou abaixado)
        if self.lista_landmark[self.landmarks_ponta_dedos[0]][1] < self.lista_landmark[self.landmarks_ponta_dedos[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        
        # Loop para todos os dedos menos o dedão
        for ponta_dedo in range(1, 5):
            # Pegar a posição no eixo Y da ponta de cada dedo
            # diminui 2 porque da ponta do dedo até o mínimo estabelecido são duas landmarks
            if (self.lista_landmark[self.landmarks_ponta_dedos[ponta_dedo]][2] < 
                self.lista_landmark[self.landmarks_ponta_dedos[ponta_dedo] - 2][2]):
                dedos.append(1)
            else:
                dedos.append(0)
        
        return dedos
    
    def encontrar_distancia(self, p1, p2, imagem, desenho=True, raio=15, espessura=3):
        x1, y1 = self.lista_landmark[p1][1:]
        x2, y2 = self.lista_landmark[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if desenho:
            cv2.line(imagem, (x1, y1), (x2, y2), (0, 0, 255), espessura)
            cv2.circle(imagem, (x1, y1), raio, (0, 0, 255), cv2.FILLED)
            cv2.circle(imagem, (x2, y2), raio, (0, 0, 255), cv2.FILLED)
            cv2.circle(imagem, (cx, cy), raio, (255, 0, 0), cv2.FILLED)
        
        comprimento = math.hypot(x2 - x1, y2 - y1)
        
        return comprimento, imagem, [x1, x2, y1, y2, cx, cy]


def main():
    tempo_anterior = 0
    tempo_atual = 0
    
    cap = cv2.VideoCapture(0)
    
    detector = DetectorMao()
    
    while True:
        sucesso, imagem = cap.read()
        imagem = detector.encontrar_maos(imagem)
        lista_landmark = detector.encontrar_posicao(imagem)
        
        # Configurar o FPS da captura
        tempo_atual = time.time()
        fps = 1 / (tempo_atual - tempo_anterior)
        tempo_anterior = tempo_atual

        # Colcar o valor de FPS na tela
        cv2.putText(imagem, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Mostrar a imagem na tela
        cv2.imshow('Imagem', imagem)
        
        # Terminar o loop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    # Fechar a tela de captura
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
