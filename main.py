import cv2

# setup classifications
age_classifications = [
    "(0-2 yrs)",
    "(4-6 yrs)",
    "(8-12 yrs)",
    "(15-20 yrs)",
    "(25-32 yrs)",
    "(38-43 yrs)",
    "(48-53 yrs)",
    "(60+ yrs)",
]
gender_classifications = ["Male", "Female"]

# define models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"

# Constants
PADDING = 15
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
FACES_DETECTION_THRESHOLD = 0.85

# load models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gender = cv2.dnn.readNet(gender_model, gender_prototxt)

# draw rectangle over faces


def draw_rectangle(faces):
    list = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > FACES_DETECTION_THRESHOLD:
            x1 = int(faces[0, 0, i, 3] * img_w)
            y1 = int(faces[0, 0, i, 4] * img_h)
            x2 = int(faces[0, 0, i, 5] * img_w)
            y2 = int(faces[0, 0, i, 6] * img_h)
            cv2.rectangle(
                img_cp, (x1, y1), (x2, y2), (0, 255, 0), int(round(img_h / 150)), 8
            )
            list.append([x1, y1, x2, y2])

    if not list:
        print("No faces detected")
        exit()
    return list


def extract_face(face_bound, img):
    face = img[
        max(0, face_bound[1] - PADDING) : min(
            face_bound[3] + PADDING, img.shape[0] - 1
        ),
        max(0, face_bound[0] - PADDING) : min(
            face_bound[2] + PADDING, img.shape[1] - 1
        ),
    ]
    return cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, True)


def generate_gender_str(blob):
    gender.setInput(blob)
    gender_prediction = gender.forward()
    return gender_classifications[gender_prediction[0].argmax()]


def generate_age_str(blob):
    age.setInput(blob)
    age_prediction = age.forward()
    return age_classifications[age_prediction[0].argmax()]


def apply_text(face_bound, img, gender_str, age_str):
    cv2.putText(
        img,
        f"{gender_str} {age_str}",
        (face_bound[0], face_bound[1] - PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


# Read Image
image = cv2.imread("imgs/sources/woman_1.jpg")
img = cv2.resize(image, (640, 720))

# Copy img
img_cp = img.copy()

# Get image dimensions and blob
img_h = img_cp.shape[0]
img_w = img_cp.shape[1]
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), [104, 117, 123], True, False)

face.setInput(blob)
detected_faces = face.forward()

# LIST OF FACES
face_bounds = draw_rectangle(detected_faces)

for face_bound in face_bounds:
    try:
        blob = extract_face(face_bound, img_cp)
        gender_str = generate_gender_str(blob)
        age_str = generate_age_str(blob)
        apply_text(face_bound, img_cp, gender_str, age_str)

    except Exception as e:
        print(e)
    continue

cv2.imshow("Result", img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
