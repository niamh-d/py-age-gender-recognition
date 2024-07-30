import cv2

#Read Image and resize
img = cv2.imread('imgs/man_1.jpg')
image = cv2.resize(img, (640, 720))

#define models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 177, 123]

#load models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gender = cv2.dnn.readNet(gender_model, gender_prototxt)

#setup classifications
age_classifications = ["0-2", "3-6", "7-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"]
gender_classifications = ["Male", "Female"]

#Copy img
img_cp = image.copy()

#Get image dimensions and blob
img_h = img_cp.shape[0]
img_w = img_cp.shape[1]
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

face.setInput(blob)
detected_faces = face.forward()

face_bounds = []

#draw rectangle over faces

for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0, 0, i, 2]
    if(confidence > 0.99):
        x1 = int(detected_faces[0, 0, i, 3] * img_w)
        x2 = int(detected_faces[0, 0, i, 4] * img_w)
        y1 = int(detected_faces[0, 0, i, 5] * img_h)
        y2 = int(detected_faces[0, 0, i, 6] * img_h)
        cv2.rectangle(img_cp, (x1, x2), (y1, y2), (0, 255, 0), int(round(img_h/150)), 8)
        face_bounds.append([x1, x2, y1, y2])


cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()


