import cv2
import sys
import numpy as np
import datetime
import os
import glob
# from retinaface import RetinaFace
from argparse import ArgumentParser
import math
import time


def check(list_ds):
    count = 0
    for i in range(len(list_ds)):
        if list_ds[i] >= list_ds[0]:
            count = count + 1
        else:
            count = count - 1
        return count


def caculate(A, B, C):
    V = [[A[0] - B[0], A[1] - B[1]], [B[1] - A[1], A[0] - B[0]]]
    b = [C[0] * (A[0] - B[0]) + C[1] * (A[1] - B[1]), A[0] * (B[1] - A[1]) + A[1] * (A[0] - B[0])]
    # A1=[[C[0]*(A[0]-B[0])+C[1]*(A[1]-B[1]),A[1]-B[1]],[A[0]*(B[1]-A[1])+A[1]*(A[0]-B[0]),A[0]-B[0]]]
    A1 = [[b[0], V[0][1]], [b[1], V[1][1]]]
    # A2=[[A[0]-B[0],C[0]*(A[0]-B[0])+C[1]*(A[1]-B[1])],[B[1]-A[1],A[1]*(B[1]-A[1])+A[1]*(A[0]-B[0])]]
    A2 = [[V[0][0], b[0]], [V[1][0], b[1]]]
    detV = np.linalg.det(np.array(V))
    detA1 = np.linalg.det(np.array(A1))
    detA2 = np.linalg.det(np.array(A2))
    x = detA1 / detV
    y = detA2 / detV
    x1 = 2 * x - C[0]
    y1 = 2 * y - C[1]
    return (x1, y1)


class pose:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-165.0, 170.0, -135.0),  # Left eye left corner
            (165.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

    def face_orientation(self, frame, landmarks):
        size = frame.shape  # (height, width, color_channel)

        image_points = np.array([
            (landmarks[2][0], landmarks[2][1]),  # Nose tip
            (landmarks[5][0], landmarks[5][1]),  # Chin
            (landmarks[0][0], landmarks[0][1]),  # Left eye left corner
            (landmarks[1][0], landmarks[1][1]),  # Right eye right corne
            (landmarks[3][0], landmarks[3][1]),  # Left Mouth corner
            (landmarks[4][0], landmarks[4][1])  # Right mouth corner
        ], dtype="double")

        # Camera internals

        center = (size[1] / 2, size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(self.model_points, rotation_vector, translation_vector, camera_matrix,
                                           dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return imgpts, modelpts, (roll, pitch, yaw), (landmarks[2][0], landmarks[2][1])
# gpuid = 0
# detector = RetinaFace('/home/doan/Desktop/insightface-master/RetinaFace/models/R50', 0, gpuid, 'net3')
# ##########################################################################
# parser = ArgumentParser()
# parser.add_argument("--video", type=str, default=None,
#                     help="Video file to be processed.")
# parser.add_argument("--cam", type=int, default=None,
#                     help="The webcam index.")
# args = parser.parse_args()
# video_src = args.cam if args.cam is not None else args.video
# if video_src is None:
#     print("Warning: video source not assigned, default webcam will be used.")
#     video_src = 0
# yaw=[]
# cap = cv2.VideoCapture(video_src)
# start_time = time.time()
#
# while True:
#     frame_got, frame = cap.read()
#     if frame_got is False:
#         break
#     if video_src == 0:
#         img = cv2.flip(frame, 2)
# ##########################################################################
#     #img = cv2.imread('/home/doan/Pictures/test_face_spoot/anhnguoi3-ir.jpeg')
#     thresh = 0.8
#     scales = [1024, 1980]
#     print(img.shape)
#     im_shape = img.shape
#     target_size = scales[0]
#     max_size = scales[1]
#     im_size_min = np.min(im_shape[0:2])
#     im_size_max = np.max(im_shape[0:2])
#     #im_scale = 1.0
#     #if im_size_min>target_size or im_size_max>max_size:
#     im_scale = float(target_size) / float(im_size_min)
#     # prevent bigger axis from being more than max_size:
#     if np.round(im_scale * im_size_max) > max_size:
#         im_scale = float(max_size) / float(im_size_max)
#
#     print('im_scale', im_scale)
#
#     scales = [im_scale]
#     flip = False
#     faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
#     end_time=time.time()
#     #landmark5 = landmarks[0].astype(np.int)
#     landmark5=landmarks[0].tolist()
#
#     x1,y1=caculate(landmark5[3],landmark5[4],landmark5[2])
#     landmark5.append([x1,y1])
#     imgpts, modelpts, rotate_degree, nose= face_orientation(img, landmark5)
#     yaw.append(rotate_degree[2])
#     print(imgpts)
#     # cv2.line(img, nose, tuple(imgpts[1].ravel().astype(np.int)), (0,255,0), 3) #GREEN
#     # cv2.line(img, nose, tuple(imgpts[0].ravel().astype(np.int)), (255,0,), 3) #BLUE
#     # cv2.line(img, nose, tuple(imgpts[2].ravel().astype(np.int)), (0,0,255), 3) #RED
#     for j in range(len(rotate_degree)):
#         cv2.putText(img, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
#     for i in range(faces.shape[0]):
#     #print('score', faces[i][4])
#         box = faces[i].astype(np.int)
#         #color = (255,0,0)
#         color = (0,0,255)
#         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#         if landmarks is not None:
#           landmark5 = np.array(landmark5).astype(np.int)
#           #print(landmark.shape)
#           for l in range(landmark5.shape[0]):
#             color = (0,0,255)
#             if l==0 or l==3:
#               color = (0,255,0)
#             cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
#     # landmark5 = np.array(landmark5).astype(np.int)
#     # color = (0,0,255)
#     # cv2.circle(img, (landmark5[5][0], landmark5[5][1]), 1, color, 2)
#     cv2.putText(img, ('{:05.2f}  seconds').format(end_time-start_time ), (200, 30 + (50 *2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
#     if (end_time-start_time == 2):
#         print(yaw)
#         count=check(yaw)
#         if count>0:
#             cv2.putText(img, 'real', (300, 30 + (50 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
#         else:
#             cv2.putText(img, 'fake', (300, 30 + (50 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
#         yaw=[]
#         start_time=end_time
#     cv2.imshow("image",img)
#     #cv2.waitKey(0)
#     if cv2.waitKey(10) == 27:
#             break
