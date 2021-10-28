import copy
import cv2
import numpy as np
import mediapipe as mp
import random
from PIL import Image
import math

koro=cv2.imread('images/koro1.jpg')
koro2 = cv2.resize(koro, dsize=(200,200))
object_size=120

rock=cv2.imread('images/rock1.png')
rock=cv2.resize(rock, dsize=(object_size,object_size))

paper=cv2.imread('images/paper1.png')
paper=cv2.resize(paper, dsize=(object_size,object_size))

item_pos=[(100,100),(200,800),(400,400),(600,100),(400,1000)]
item_size=100
ball=cv2.imread('images/ball.png')
ball=cv2.resize(ball, dsize=(item_size,item_size))

global score

Hands = mp.solutions.hands
Draw = mp.solutions.drawing_utils

def putSprite_mask(back, front4, pos):
    y, x = pos
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x+fw, bw), min(y+fh, bh)
    if not ((-fw < x < bw) and (-fh < y < bh)) :
        return back
    front3 = front4[:, :, :3]
    front_roi = front3[y1-y:y2-y, x1-x:x2-x]
    roi = back[y1:y2, x1:x2]
    tmp = np.where(front_roi==(0,0,0), roi, front_roi)
    back[y1:y2, x1:x2] = tmp
    return back

class HandDetector:
    def __init__(self, max_num_hands=12, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = Hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)
    
    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image
        show_image=originalImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB
        results = self.hands.process(image)
        
        image
        back=cv2.imread('images/back.png')
        back=cv2.resize(back, dsize=(1280,720))
        #kata
        #gameimg=copy.deepcopy(back)
        
        
        global score
        gameimg=back
        for pos in item_pos:
            putSprite_mask(gameimg,ball,pos)
            #gameimg[pos[0]:pos[0]+item_size,pos[1]:pos[1]+item_size]=ball
        if results.multi_handedness:
            label = results.multi_handedness[handNumber].classification[0].label  # label gives if hand is left or right
            #account for inversion in cam
            if label == "Left":
                label = "Right"
            elif label == "Right":
                label = "Left"
        if results.multi_hand_landmarks:
            n=0
            
            for hand in results.multi_hand_landmarks:  # returns None if hand is not found
                #hand = results.multi_hand_landmarks[handNumber] #results.multi_hand_landmarks returns landMarks for all the hands
                landMarkList = []
                for id, landMark in enumerate(hand.landmark):
                    # landMark holds x,y,z ratios of single landmark
                    imgH, imgW, imgC = originalImage.shape  # height, width, channel for image
                    
                    xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                    landMarkList.append([id, xPos, yPos, label])

                if draw:
                    Draw.draw_landmarks(originalImage, hand, Hands.HAND_CONNECTIONS)
                #print('Handedness:', results.multi_handedness)
                count=0
                x=0
                y=0
                if(len(landMarkList) != 0):
                    # handLandmarks[point of 21 points][x or y] locates finger positions.
                    # see details: https://google.github.io/mediapipe/solutions/hands
                    # handLandmarks[4][1] 4->Thumb_tip 1->x-axis
                    # handLandmarks[8][2] 8->Index_finger_tip 2->y-axis

                    if landMarkList[4][1]+50 < landMarkList[5][1]:       #Thumb finger
                        count = count+1
                    if landMarkList[7][2] < landMarkList[5][2]:       #Index finger
                        count = count+1
                    if landMarkList[11][2] < landMarkList[9][2]:     #Middle finger
                        count = count+1
                    if landMarkList[15][2] < landMarkList[13][2]:     #Ring finger
                        count = count+1
                    if landMarkList[19][2] < landMarkList[17][2]:     #Little finger
                        count = count+1
                    x=landMarkList[4][1]
                    y=landMarkList[4][2]
                    #cv2.putText(originalImage, str(landMarkList[11][2]), (landMarkList[11][1] , landMarkList[11][2] ), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
                    #cv2.putText(originalImage, str(landMarkList[9][2]), (landMarkList[9][1], landMarkList[9][2]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 25)
                    #print(count)
                cv2.putText(show_image, str(count), (45, 275+n), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
                n=n+50
                landmark_x=x-100
                landmark_y=y-100
                if count > 3:
                    putSprite_mask(gameimg,paper,(landmark_y,landmark_x))
                    #gameimg[max(landmark_y,0):max(landmark_y,0)+min(gameimg.shape[0]-landmark_y,object_size),max(landmark_x,0):max(landmark_x,0)+min(gameimg.shape[1]-landmark_x,object_size)]=paper[:min(gameimg.shape[0]-landmark_y,object_size),:min(gameimg.shape[1]-landmark_x,object_size)]            #image[landmark_x]
                else :
                    putSprite_mask(gameimg,rock,(landmark_y,landmark_x))
                    #gameimg[max(landmark_y,0):max(landmark_y,0)+min(gameimg.shape[0]-landmark_y,object_size),max(landmark_x,0):max(landmark_x,0)+min(gameimg.shape[1]-landmark_x,object_size)]=rock[:min(gameimg.shape[0]-landmark_y,object_size),:min(gameimg.shape[1]-landmark_x,object_size)]            #image[landmark_x]
                    i=0
                    for pos in item_pos:
                        if pos[1]<landmark_x+100 and pos[1]+item_size>landmark_x and pos[0]<landmark_y+100 and pos[0]+item_size>landmark_y:
                            
                            score=score+100
                            item_pos[i]=(random.randint(0,600),random.randint(0,1160))
                        i=i+1
                
                #cv2.putText(show_image, "x="+str(x), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                #cv2.putText(show_image, "y="+str(y), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        return gameimg



def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]



def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    back=cv2.imread('images/back.png')
    back=cv2.resize(back, dsize=(1280,720))
    #kata
    #gameimg=copy.deepcopy(back)
    
    
    global score
    gameimg=back
    for pos in item_pos:
        gameimg[pos[0]:pos[0]+item_size,pos[1]:pos[1]+item_size]=ball
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        
        if index == 19:  # 右手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 30, (0, 255, 0), 2)
            landmark_x=landmark_x-100
            landmark_y=landmark_y-100
            gameimg[max(landmark_y,0):max(landmark_y,0)+min(gameimg.shape[0]-landmark_y,200),max(landmark_x,0):max(landmark_x,0)+min(gameimg.shape[1]-landmark_x,200)]=object[:min(gameimg.shape[0]-landmark_y,200),:min(gameimg.shape[1]-landmark_x,200)]            #image[landmark_x]
            i=0
            for pos in item_pos:
                if pos[1]<landmark_x+100 and pos[1]+item_size>landmark_x and pos[0]<landmark_y+100 and pos[0]+item_size>landmark_y:
                    
                    score=score+100
                    item_pos[i]=(random.randint(0,600),random.randint(0,1160))
                i=i+1
                    

        if index == 20:  # 左手2(先端)
            landmark_x=landmark_x-100
            landmark_y=landmark_y-100
            gameimg[max(landmark_y,0):max(landmark_y,0)+min(gameimg.shape[0]-landmark_y,200),max(landmark_x,0):max(landmark_x,0)+min(gameimg.shape[1]-landmark_x,200)]=object[:min(gameimg.shape[0]-landmark_y,200),:min(gameimg.shape[1]-landmark_x,200)]            #image[landmark_x]
            i=0
            for pos in item_pos:
                if pos[1]<landmark_x+100 and pos[1]+item_size>landmark_x and pos[0]<landmark_y+100 and pos[0]+item_size>landmark_y:
                    
                    score=score+100
                    item_pos[i]=(random.randint(0,600),random.randint(0,1160))
                i=i+1
        
        # if not upper_body_only:
        if True:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv2.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv2.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv2.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv2.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv2.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv2.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv2.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv2.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv2.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv2.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv2.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv2.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv2.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv2.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv2.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv2.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv2.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv2.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv2.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv2.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv2.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv2.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv2.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv2.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return gameimg


    
def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append([landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    
    plt.pause(.001)

    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image
handDetector = HandDetector(min_detection_confidence=0.7)

def main():
    print("capturing...")
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    print("captured")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1140)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("set")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        # upper_body_only=upper_body_only,
        # model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    global score
    score=0
    lastimg=back=cv2.imread('images/back.png')
    lastimg=cv2.resize(lastimg, dsize=(1280,720))
    


    while True:
        #display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        
        if not ret:
            print("notcupture")
            break
        image = cv2.flip(image, 1)  # ミラー表示
        img2 = handDetector.findHandLandMarks(image=image, draw=True)
        debug_image = copy.deepcopy(image)
        ca = copy.deepcopy(image)
        # # 検出実施 #############################################################
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = pose.process(image)
        # if results.pose_landmarks is not None:
        #     # 外接矩形の計算
        #     brect = calc_bounding_rect(debug_image, results.pose_landmarks)
        #     # 描画
        #     debug_image = draw_landmarks(
        #         debug_image,
        #         results.pose_landmarks,
        #         # upper_body_only,
        #     )
        #     debug_image = draw_bounding_rect(False, debug_image, brect)
        # else:
        #     debug_image=copy.deepcopy(lastimg)
            
        
        
        cv2.putText(img2,'score:',(5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(img2,str(score),(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        #cv2.imshow('camera', ca)
        #cv2.imshow('MediaPipe Pose Demo', debug_image)
        cv2.imshow("result", img2)
        cv2.moveWindow("result",200,200)
        cv2.setWindowProperty("result", cv2.WND_PROP_TOPMOST, 1)    
        lastimg=debug_image

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
    return 100



if __name__ == '__main__':
    main()