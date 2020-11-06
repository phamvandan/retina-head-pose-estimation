import cv2
from detect import load_net, do_detect
from east import caculate, pose
import numpy as np
import random
from configparser import ConfigParser


def read_config():
    config = ConfigParser()
    config.read("state.cfg")
    return config


class straight:
    def __init__(self, config):
        self.min_yaw = float(config["straight"]["min_yaw"])
        self.max_yaw = float(config["straight"]["max_yaw"])


class left:
    def __init__(self, config):
        self.max_yaw = float(config["left"]["max_yaw"])


class para:
    def __init__(self):
        self.config = read_config()
        self.max_thresh = int(self.config["thresh"]["max_count"])
        self.max_reset_count = int(self.config["thresh"]["max_reset_count"])
        self.straight = straight(self.config)
        self.left = left(self.config)


def convert_landmark(landms):
    landmark = []
    i = 0
    while i < 10:
        landmark.append([landms[i], landms[i + 1]])
        i = i + 2
    x1, y1 = caculate(landmark[3], landmark[4], landmark[2])
    landmark.append([x1, y1])
    return landmark


def draw_axis(img, imgpts, nose):
    cv2.line(img, nose, tuple(imgpts[1].ravel().astype(np.int)), (0, 255, 0), 3)  # GREEN
    cv2.line(img, nose, tuple(imgpts[0].ravel().astype(np.int)), (255, 0,), 3)  # BLUE
    cv2.line(img, nose, tuple(imgpts[2].ravel().astype(np.int)), (0, 0, 255), 3)  # RED


def draw(img, rotate_degree, dets, landmark):
    for j in range(len(rotate_degree)):
        name = "roll"
        if j == 1:
            name = "pitch"
        if j == 2:
            name = "yaw"
        cv2.putText(img, ('{:05.2f}').format(float(rotate_degree[j])) + "_" + name, (10, 30 + (50 * j)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
    # print('score', faces[i][4])
    box = dets[0].astype(np.int)
    # color = (255,0,0)
    color = (0, 0, 255)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    landmark = np.array(landmark).astype(np.int)
    # print(landmark.shape)
    for l in range(landmark.shape[0]):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(img, (landmark[l][0], landmark[l][1]), 1, color, 2)

def check_straight(img, param, count, state_index, reset_count ):
    cv2.putText(img, "nhin thang", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    if param.straight.min_yaw < yaw < param.straight.max_yaw:
        count = count + 1
        cv2.putText(img, str(count), (300, 30 + (50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        if count == param.max_thresh:
            count = 0
            state_index += 1
            cv2.putText(img, "PASSED", (300, 30 + (30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        reset_count = 0
    else:
        reset_count += 1
        if reset_count == param.max_reset_count:
            count = 0
            reset_count = 0
            cv2.putText(img, "RESET", (300, 30 + (30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    return count, state_index, reset_count

def check_left(img, param, count, state_index, reset_count ):
    cv2.putText(img, "quay trai", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    if param.left.max_yaw > yaw:
        count = count + 1
        cv2.putText(img, str(count), (300, 30 + (50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        if count == param.max_thresh:
            count = 0
            state_index += 1
            cv2.putText(img, "PASSED", (300, 30 + (30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        reset_count = 0
    else:
        reset_count += 1
        if reset_count == param.max_reset_count:
            count = 0
            reset_count = 0
            cv2.putText(img, "RESET", (300, 30 + (30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    return count, state_index, reset_count


## check state
# 0. nhin thang
# 1. quay sang trai
# 2. ngua mat len tren
# 3. quay sang phai
# 4. up mat xuong duoi

if __name__ == '__main__':
    pose_estimator = pose()
    net, device, cfg = load_net()
    param = para()
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    write_video = False
    if write_video:
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    rolls, pitches, yaws = [], [], []

    check_state = [0, 1, 2, 3, 4]
    # random.shuffle(check_state)
    print(check_state)
    count = 0
    state_index = 0
    reset_count = 0
    while True:
        ret, img = cap.read()
        dets, landms = do_detect(img, net, device, cfg)
        if dets is None or landms is None or len(dets) == 0 or len(landms) == 0:
            continue
        landms = landms[0]
        landmark = convert_landmark(landms)
        imgpts, modelpts, rotate_degree, nose = pose_estimator.face_orientation(img, landmark)
        roll, pitch, yaw = rotate_degree

        if check_state[state_index] == 0:  ## nhin thang
            count, state_index, reset_count = check_straight(img, param, count, state_index, reset_count)
        elif check_state[state_index] == 1:  ## quay trai
            count, state_index, reset_count = check_left(img, param, count, state_index, reset_count)

        # print(imgpts)
        draw_axis(img, imgpts, nose)
        draw(img, rotate_degree, dets, landmark)

        # landmark5 = np.array(landmark5).astype(np.int)
        # color = (0,0,255)
        # cv2.circle(img, (landmark5[5][0], landmark5[5][1]), 1, color, 2)
        # cv2.putText(img, ('{:05.2f}  seconds').format(end_time-start_time ), (200, 30 + (50 *2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        # if (end_time-start_time == 2):
        #     print(yaw)
        #     count=check(yaw)
        #     if count>0:
        #         cv2.putText(img, 'real', (300, 30 + (50 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        #     else:
        #         cv2.putText(img, 'fake', (300, 30 + (50 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        #     yaw=[]
        #     start_time=end_time
        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:
            cv2.waitKey(0)
        if write_video:
            out.write(img)

    if write_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    #
