import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('Capture - Face detection', frame)

ESC_KEYCODE = 27

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame)

        if cv2.waitKey(10) == ESC_KEYCODE:
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()