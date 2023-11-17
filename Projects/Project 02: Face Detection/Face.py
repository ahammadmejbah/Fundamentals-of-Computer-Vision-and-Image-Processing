import cv2
import numpy as np

network_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

capture_image = cv2.VideoCapture(0)

while True:
    ret, frame = capture_image.read()
    if not ret:
        break
    # Image Value = 0, 255
    # 0/255 = 0 min value
    # 255/255 = 1 max value
    image_resize = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 117, 123))
    network_model.setInput(image_resize)
    face_detection = network_model.forward() # Weight and Bias

    for i in range(face_detection.shape[2]):
        detection_confidence = face_detection[0, 0, i, 2]

        if detection_confidence > 0.5:
            bounding_box = face_detection[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x_start, y_start, x_end, y_end) = bounding_box.astype(int)

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            text_data = f"Face Detection With {detection_confidence:.2f}"

            cv2.putText(frame, text_data, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_image.release()
cv2.destroyAllWindows()
