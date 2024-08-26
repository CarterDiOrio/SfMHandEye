"""
A utility detecting charuco chessboards and outputing the corners in the mrcal compatible 
corners.vln file
"""
import argparse
import cv2
import glob
import cv2.aruco
from typing import Dict, Optional, List
import numpy as np
from functools import partial
import mecademicpy.robot as mdr
from scipy.spatial.transform import Rotation as R
import random

class Detection:
  filename: str = ""
  detected_corners: Dict[int, np.ndarray] = {}
  
# K = np.array([
#    [1381.17626953125, 0, 973.329956054688], 
#    [0, 1381.80151367188, 532.698852539062], 
#    [0, 0, 1]
# ])

# T_hand_eye = np.array([
#   [-0.00328758,    0.999953,  -0.0091034,     -21.068],
#   [-0.999881, -0.00342452,  -0.0150681,     31.6416],
#  [-0.0150986,  0.00905277,    0.999845,     26.6441],
#           [0,           0,           0,           1]
# ])

# load mapx and mapy from mapx.yml in home directory
maps = cv2.FileStorage("/home/cdiorio/mapx.yml", cv2.FILE_STORAGE_READ)
mapx = maps.getNode("mapx").mat()
mapy = maps.getNode("mapy").mat()

def detect(dictionary: cv2.aruco.Dictionary,
           board: cv2.aruco.Board, 
           detector_params: cv2.aruco.DetectorParameters, 
           img: cv2.Mat) -> Optional[Detection]:

  marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
    img, 
    dictionary,
    parameters=detector_params
  )

  
  if marker_ids is not None and  len(marker_ids) > 0:
    cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)

    if ret:
      cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

      obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
      ret = cv2.solvePnPRansac(obj_points, img_points, K, None)
      if (ret):
        rvec = ret[1]
        tvec = ret[2]
        rvec, tvec = cv2.solvePnPRefineLM(obj_points, img_points, K, None, rvec, tvec, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))

        rotation, _ = cv2.Rodrigues(rvec)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = rotation
        T_cam_board[:3, 3] = np.atleast_2d(tvec).T * 1000

        return T_cam_board
      else:
        print(ret)


def main():
  
  robot = mdr.Robot()
  robot.Connect(address='192.168.0.100', enable_synchronous_mode=True)

  robot.ResetError()
  robot.ActivateRobot()
  robot.Home()
  robot.WaitActivated()
  robot.WaitHomed()

  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
  board = cv2.aruco.CharucoBoard((24, 17), 0.030, 0.022, dictionary)
  params = cv2.aruco.DetectorParameters()
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  params.cornerRefinementMinAccuracy = 0.01
  params.cornerRefinementMaxIterations = 1000


  cap = cv2.VideoCapture(6)

  # set camera to 1920 x 1080
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_FPS, 30)

  for i in range(0, 30):
    ret, img = cap.read()
    cv2.imshow("image", img)
    cv2.waitKey(1)


  robot.MoveJoints(-90, 0, 0, 0, 0, 0)
  pose = robot.GetRtTargetCartPos()
  r = R.from_euler('XYZ', pose[3:], True)
  T_world_hand = np.eye(4)
  T_world_hand[:3, :3] = r.as_matrix()
  T_world_hand[:3, 3] = pose[:3]


  camera_positions = []
  hand_positions = []
  c = 0
  run = True
  while run and c < 100:
    x = random.uniform(-20, 20)
    y = random.uniform(-60, 60)
    z = random.uniform(-20, 0)
    roll = random.uniform(-25, 25)
    pitch = random.uniform(-25, 25)
    yaw = random.uniform(-30, 30)
    r = R.from_euler('XYZ', [roll, pitch, yaw], degrees=True)
    T_hand_offset = np.eye(4)
    T_hand_offset[0, 3] = x
    T_hand_offset[1, 3] = y
    T_hand_offset[2, 3] = z
    T_hand_offset[:3, :3] = r.as_matrix()

    # apply offset to hand
    desired_pose = T_world_hand @ T_hand_offset
    xyz = R.from_matrix(desired_pose[:3, :3]).as_euler("XYZ", degrees=True)
    robot.MovePose(desired_pose[0, 3], desired_pose[1, 3], desired_pose[2, 3], *xyz)
    
    T_camera_board = None
    for i in range(0, 15):
      ret, img = cap.read()

      # remap image
      img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

      try:
        T_camera_board = detect(dictionary, board, params, img)
      except:
        pass
      cv2.imshow("image", img)
      if cv2.waitKey(1) == ord('q'):
        run = False
        break

    if (T_camera_board is not None):
      print("added pose: ", len(camera_positions))
      camera_positions.append(T_camera_board)
      hand_positions.append(robot.GetRtTargetCartPos(False, True))

    c += 1

  robot.WaitIdle()
  # robot.DeactivateRobot()
  robot.Disconnect()

  for idx, pose in enumerate(hand_positions):
    r = R.from_euler('XYZ', pose[3:], True)
    T_world_hand = np.eye(4)
    T_world_hand[:3, :3] = r.as_matrix()
    T_world_hand[:3, 3] = pose[:3]
    hand_positions[idx] = T_world_hand

  average_translation_err = 0
  average_rotation_err = 0
  count = 0
  for idx1, pose1 in enumerate(camera_positions):
    for idx2, pose2 in enumerate(camera_positions):
      if (idx1 == idx2):
        continue
      
      T_camera1_camera2 = pose1 @ np.linalg.inv(pose2)
      T_hand1_hand2 = np.linalg.inv(hand_positions[idx1]) @ hand_positions[idx2]

      T_camera1_hand2 = T_camera1_camera2 @ np.linalg.inv(T_hand_eye)
      T_camera1_hand2_hat = np.linalg.inv(T_hand_eye) @ T_hand1_hand2

      T_err = T_camera1_hand2 @ np.linalg.inv(T_camera1_hand2_hat)

      count += 1

      average_translation_err += np.linalg.norm(T_err[:3, 3])
      average_rotation_err += np.linalg.norm(R.from_matrix(T_err[:3, :3]).as_euler('XYZ', True))

  print(average_translation_err / count)
  print(average_rotation_err / count)
  


if __name__ == "__main__":
  main()
