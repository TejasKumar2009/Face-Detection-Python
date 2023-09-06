import cv2 as cv
import mediapipe as mp

capture = cv.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    isTrue, frame = capture.read()

    RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = faceDetection.process(frame)

    if (results.detections):
        for face in list(results.detections):

            h, w, c = frame.shape

            rbb = face.location_data.relative_bounding_box
            face_points = int(rbb.xmin*w), int(rbb.ymin*h), int(rbb.width*w), int(rbb.height*h)

            face_score = int(face.score[0]*100)

            if (face_score>=90):
                cv.rectangle(frame, face_points, (0,255,0), 3)
                cv.putText(frame, f"{face_score}%", (face_points[0], face_points[1]-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            elif (face_score>=70):
                cv.rectangle(frame, face_points, (255,0,0), 3)
                cv.putText(frame, f"{face_score}%", (face_points[0], face_points[1]-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
            else:
                cv.rectangle(frame, face_points, (0,0,255), 3)
                cv.putText(frame, f"{face_score}%", (face_points[0], face_points[1]-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)


    cv.imshow("Face Detection", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

capture.release()
cv.destroyAllWindows()
