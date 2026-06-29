from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import time
import math

try:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import pose_landmarker as mp_pose
except ImportError:
    mp_vision = None
    base_options_lib = None
    mp_pose = None


_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.using_tasks_api = not hasattr(mp, "solutions")

        if self.using_tasks_api:
            self.mpDraw = mp_vision.drawing_utils
            self.mpPose = mp_pose
            model_path = self._ensure_pose_model()
            options = self.mpPose.PoseLandmarkerOptions(
                base_options=base_options_lib.BaseOptions(
                    model_asset_path=model_path
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.detectionCon,
                min_pose_presence_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon,
            )
            self.pose = self.mpPose.PoseLandmarker.create_from_options(options)
        else:
            self.mpDraw = mp.solutions.drawing_utils
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                         self.detectionCon, self.trackCon)

    def _ensure_pose_model(self):
        model_path = Path(__file__).resolve().parent / "data" / "pose_landmarker_full.task"
        if model_path.exists():
            return str(model_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(_POSE_MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "MediaPipe 0.10+ requires a pose landmarker model file. "
                f"Tried to download it to {model_path}, but the download failed."
            ) from exc
        return str(model_path)

    def _current_landmarks(self):
        if not self.results:
            return []

        if self.using_tasks_api:
            return self.results.pose_landmarks[0] if self.results.pose_landmarks else []

        return self.results.pose_landmarks.landmark if self.results.pose_landmarks else []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.using_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
            self.results = self.pose.detect(mp_image)
            if self.results.pose_landmarks and draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks[0],
                    self.mpPose.PoseLandmarksConnections.POSE_LANDMARKS,
                )
        else:
            self.results = self.pose.process(imgRGB)

            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                               self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        landmarks = self._current_landmarks()
        if landmarks:
            for id, lm in enumerate(landmarks):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) 
                           - math.atan2(y1-y2, x1-x2))
        
        if angle < 0:
            angle += 360

        #print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle