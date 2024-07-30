import cv2

# Read Image and resize
img = cv2.imread("imgs/girl_1.jpg")
image = cv2.resize(img, (640, 720))

# define models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 177, 123]

# load models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gender = cv2.dnn.readNet(gender_model, gender_prototxt)

# setup classifications
age_classifications = [
    "(0-2 yrs)",
    "(4-6 yrs)",
    "(8-12 yrs)",
    "(15-20 yrs)",
    "(25-32 yrs)",
    "(35-43 yrs)",
    "(48-53 yrs)",
    "(60+ yrs)",
]
gender_classifications = ["Male", "Female"]

# Copy img
img_cp = image.copy()

# Get image dimensions and blob
img_h = img_cp.shape[0]
img_w = img_cp.shape[1]
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

face.setInput(blob)
detected_faces = face.forward()

face_bounds = []

# draw rectangle over faces

for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0, 0, i, 2]
    if confidence > 0.99:
        x1 = int(detected_faces[0, 0, i, 3] * img_w)
        y1 = int(detected_faces[0, 0, i, 4] * img_h)
        x2 = int(detected_faces[0, 0, i, 5] * img_w)
        y2 = int(detected_faces[0, 0, i, 6] * img_h)
        cv2.rectangle(
            img_cp, (x1, y1), (x2, y2), (0, 255, 0), int(round(img_h / 150)), 8
        )
        face_bounds.append([x1, y1, x2, y2])

if not face_bounds:
    print("No faces detected")
    exit()

for face_bound in face_bounds:
    try:
        face = img_cp[
            max(0, face_bound[1] - 15) : min(face_bound[3] + 15, img_cp.shape[0] - 1),
            max(0, face_bound[0] - 15) : min(face_bound[2] + 15, img_cp.shape[1] - 1),
        ]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

        gender.setInput(blob)
        gender_prediction = gender.forward()
        gender_str = gender_classifications[gender_prediction[0].argmax()]

        age.setInput(blob)
        age_prediction = age.forward()
        age_str = age_classifications[age_prediction[0].argmax()]

        cv2.putText(
            img_cp,
            f"{gender_str} {age_str}",
            (face_bound[0], face_bound[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    except Exception as e:
        print(e)
    continue

cv2.imshow("Result", img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
