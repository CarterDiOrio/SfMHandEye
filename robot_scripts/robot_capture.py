import mecademicpy.robot as mdr
import cv2
import time
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

def main(address: str):
  robot = mdr.Robot()
  robot.Connect(address=address, enable_synchronous_mode=True)

  robot.ResetError()
  robot.ActivateRobot()
  robot.Home()
  robot.WaitActivated()
  robot.WaitHomed()
  
  print(robot.GetStatusRobot())


  print(robot.GetPose())


  y_range = 100
  z_range = 20

  pose = robot.GetPose()

  cap = cv2.VideoCapture(4)


  # set camera to 1920 x 1080
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


  # warm up camera
  for i in range(0, 30):
    ret, a = cap.read()
    cv2.imshow("image", a)
    cv2.waitKey(1)



  calibration_data = {
    "groups": []
  }

  image_id = 0

  # first group of cameras is of known translation
  robot.MoveJoints(60, 0, 0, 0, 0, 0)
  pose = robot.GetRtTargetCartPos()
  r = R.from_euler('X', pose[3], degrees=True) * R.from_euler('Y', pose[4], degrees=True) * R.from_euler('Z', pose[5], degrees=True)
  T_world_hand = np.eye(4)
  T_world_hand[:3, :3] = r.as_matrix()
  T_world_hand[:3, 3] = pose[:3]

  group = {
        "cameras": []
      }

  for c in range(-50, 50 + 10, 10):
    T_hand_offeset = np.eye(4)
    T_hand_offeset[0, 3] = 0
    T_hand_offeset[1, 3] = c

    # apply offset to hand
    desired_pose = T_world_hand @ T_hand_offeset

    print(robot.GetPose())
    xyz = R.from_matrix(desired_pose[:3, :3]).as_euler("XYZ", degrees=True)

    robot.MovePose(desired_pose[0, 3], desired_pose[1, 3], desired_pose[2, 3], *xyz)

    for i in range(0, 22):
        ret, frame = cap.read()
        cv2.imshow("image", frame)
        cv2.waitKey(1)

    name = f"{image_id}.png"
    cv2.imwrite(f"./data/{name}", frame)
    image_id += 1

    group["cameras"].append({
      "pose": robot.GetRtTargetCartPos(False, True),
      "image": name
    })
  
  calibration_data["groups"].append(group)


  joint_ranges = [
    [0, 60],
    [-25, 25],
    [-25, 25],
    [-25, 25],
    [-45, 45],
    [-45, 45],
  ]

  # get random images
  for i in range(0, 1000):


      joints = [random.uniform(*r) for r in joint_ranges]

      group = {
        "cameras": []
      }


      robot.MoveJoints(*joints)
      
      pose = robot.GetRtTargetCartPos(False, True)


      for i in range(0, 22):
        ret, frame = cap.read()
        cv2.imshow("image", frame)
        cv2.waitKey(1)

      name = f"{image_id}.png"
      cv2.imwrite(f"./data/{name}", frame)
      image_id += 1

      group["cameras"].append({
        "pose": pose,
        "image": name
      })
  
      calibration_data["groups"].append(group)

    
  with open("./data/data.json", "w") as file:
    json.dump(calibration_data, file, indent=4)



  robot.WaitIdle()
  robot.DeactivateRobot()
  robot.Disconnect()


    

if __name__ == "__main__":
  main('192.168.0.100')