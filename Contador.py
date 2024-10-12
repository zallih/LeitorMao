import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Pontos das mãos
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Cores
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = "_"
        thickness = [2, 2, 2, 2, 2]
        
        if results.multi_hand_landmarks:
            coordinates_thumb = []
            coordinates_palm = []
            coordinates_ft = []
            coordinates_fb = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Coleta de coordenadas
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])
                
                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])
                
                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])
                
                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])
                
                # Calcular ângulo do polegar
                if len(coordinates_thumb) == 3:
                    p1 = np.array(coordinates_thumb[0])
                    p2 = np.array(coordinates_thumb[1])
                    p3 = np.array(coordinates_thumb[2])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    thumb_finger = angle > 150
                else:
                    thumb_finger = False

                # Inicializa a contagem de dedos
                fingers = np.zeros(5, dtype=bool)
                fingers[0] = thumb_finger  # Estado do polegar

                # Calcular se os outros dedos estão levantados
                for i in range(4):  # Para os dedos 1 a 4 (índice a mindinho)
                    tip = np.array(coordinates_ft[i])
                    base = np.array(coordinates_fb[i])
                    
                    # Adiciona verificação para considerar levantado somente se a ponta estiver acima da base
                    if tip[1] < base[1] - 30:  # Considera levantado se a ponta estiver acima da base
                        fingers[i + 1] = True  # Dedo levantado
                    else:
                        fingers[i + 1] = False  # Dedo fechado
                
                # Verifica se todos os dedos estão fechados
                if not np.any(fingers):
                    fingers_counter = "0"
                else:
                    fingers_counter = str(np.count_nonzero(fingers))

                for (i, finger) in enumerate(fingers):
                    if finger:
                        thickness[i] = -1
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        ################################
        # Visualização
        cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)
        
        # Pulgar
        cv2.rectangle(frame, (100, 10), (150, 60), PEACH, thickness[0])
        cv2.putText(frame, "Polegar", (100, 80), 1, 1, (255, 255, 255), 2)
        
        # Índice
        cv2.rectangle(frame, (160, 10), (210, 60), PURPLE, thickness[1])
        cv2.putText(frame, "Indicador", (160, 80), 1, 1, (255, 255, 255), 2)
        
        # Medio
        cv2.rectangle(frame, (220, 10), (270, 60), YELLOW, thickness[2])
        cv2.putText(frame, "Medio", (220, 80), 1, 1, (255, 255, 255), 2)
        
        # Anular
        cv2.rectangle(frame, (280, 10), (330, 60), GREEN, thickness[3])
        cv2.putText(frame, "Anelar", (280, 80), 1, 1, (255, 255, 255), 2)
        
        # Menique
        cv2.rectangle(frame, (340, 10), (390, 60), BLUE, thickness[4])
        cv2.putText(frame, "Mindinho", (340, 80), 1, 1, (255, 255, 255), 2)
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
