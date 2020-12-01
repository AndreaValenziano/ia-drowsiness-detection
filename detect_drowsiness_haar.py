# Utilizzo:
# python detect_drowsiness_haar.py --alarm alarm.wav


from threading import Thread
import playsound

import cv2


def contains(rect1, rect2):
    # Coordinate primo rettangolo
    x1 = rect1[0]
    y1 = rect1[1]
    w1 = rect1[2]
    h1 = rect1[3]

    # Coordinate secondo rettangolo
    x2 = rect2[0]
    y2 = rect2[1]
    w2 = rect2[2]
    h2 = rect2[3]

    # verifica se è contenuto prima orizzontalmente (larghezza del secondo rettangolo inferiore al primo)
    # e poi verticalmente: altezza del secondo minore del primo rettangolo
    return x1 < x2 < x2 + w2 < x1 + w1 and y1 < y2 < y2 + h2 < y1 + h1


def sound_alarm(path):
    # Fa partire l' allarme
    playsound.playsound(path)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
alarm = 'alarm.wav'

# definire una costante per indicare il massimo di frame consecutivi
# in cui posso mancare gli occhi dal volto (occhi chiusi)
EYE_AR_CONSEC_FRAMES = 8

# inizializza il contatore dei fotogrammi
# e un valore booleano utilizzato per indicare se l'allarme si sta attivando
COUNTER = 0
ALARM_ON = False
print("Starting video stream thread...")

cap = cv2.VideoCapture(0)
if not cap.read()[0]:
    print("Webcam non è disponibile")
    exit(0)
# Iniziare un ciclo fin tanto che è aperta la cattura del video
while cap.isOpened():
    _, frame = cap.read()
    # conversione in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # recuperare il volto e gli occhi tramite haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), )

    # disegnare un rettangolo per identificare i volti
    for f in faces:
        cv2.rectangle(frame, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 2)
        for e in eyes:
            if contains(f, e):
                # disegnare due rettangoli per identificare gli occhi se essi sono contenuti nel volto
                cv2.rectangle(frame, (e[0], e[1]), (e[0] + e[2], e[1] + e[3]), (255, 0, 0), 2)

        # verificare se le all'interno di un volto sono assenti gli occhi
        # se sì: aumentare il contatore
        if len(eyes) == 0:
            COUNTER += 1

            # In caso di superamento della soglia far suonare l'allarme
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                    # avviare un thread con il suono dell'allarme in riproduzione
                    t = Thread(target=sound_alarm, args=(alarm,))
                    t.deamon = True
                    t.start()

                # Segnalare l'allarme a video sul frame
                cv2.putText(frame, "ALLARME SONNOLENZA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # se invece sono presenti degli occhi nel volto
            # -> azzerare il contatore e spegnere l'allarme
        else:
            COUNTER = 0
            ALARM_ON = False

    # Indicare a video
    # le soglie e i contatori dei frame per rendere più semplice il debug
    cv2.putText(frame, "COUNTER: %s" % COUNTER, (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # mostrare il frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Preme q per interrompere il video stream
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
