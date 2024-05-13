import torch
import cv2
import pandas as pd

from ultralytics import YOLO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIDEO_PATH = r"C:\Users\macaron\Desktop\VISION_LAB\홍성민_작업폴더\DonggukGraduated\낙상사고.mp4"
OUTPUT_PATH = r"C:\Users\macaron\Desktop\VISION_LAB\홍성민_작업폴더\DonggukGraduated\output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print('='*20)
    print('No Cap')
    print('='*20)
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'), 
    frame_rate,
    (frame_width, frame_height)
)

model = YOLO('yolov8m-pose.pt')
model.to(DEVICE)

keypoint_names = ['Nose_width', 'Noise_height',
                  'Right Eye width', 'Right Eye height',
                  'Left Eye_width', 'Left Eye_height',
                  'Right Ear_width', 'Right Ear_height',
                  'Left Ear_width', 'Left Ear_height',
                  'Right Shoulder_width', 'Right Shoulder_height', 
                  'Left Shoulder_width', 'Left Shoulder_height',
                  'Right Elbow_width', 'Right Elbow_height',
                  'Left Elbow_width', 'Left Elbow_height',
                  'Right Wrist_width', 'Right Wrist_height',
                  'Left Wrist_width', 'Left Wrist_height', 
                  'Right Hip_width', 'Right Hip_height',
                  'Left Hip_width', 'Right Hip_height',
                  'Right Knee_width', 'Right Knee_height',
                  'Left Knee_width', 'Left Knee_height',
                  'Right Ankle_width', 'Right Ankle_height',
                  'Left Ankle_width', 'Right Ankle_height'
                  ]

# DataFrame 생성
columns = ['Frame']
columns.extend([f'{name}' for name in keypoint_names])
df = pd.DataFrame(columns=columns)
frame_number = 0

def bbox_info(results, box, tracking=False):
    left, top = int(box[0]), int(box[1])
    right, bottom = int(box[2]), int(box[3])

    if tracking:
        conf, label_name = box[5], results[0].names[int(box[6])]

    else:
        conf, label_name = box[4], results[0].names[int(box[5])]

    return left, top, right, bottom, conf, label_name

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('='*20)
        print('No Frame')
        print('='*20)
        break

    results = model.track(frame,
                          conf=0.5,
                          persist=True,
                          verbose=False)

    boxes = results[0].boxes.cpu().numpy().data
    keypoints = results[0].keypoints
    keypoints_info = keypoints.xy[0].data.cpu().numpy().flatten()

    if len(boxes) != 0:
        # BBox
        for box in boxes:
            # 관절
            # tracking (O)
            if len(box) == 6:
                left, top, right, bottom, conf, label_name = bbox_info(results, box)

            # tracking (X)
            elif len(box) == 7:
                left, top, right, bottom, conf, label_name = bbox_info(results, box)

        # 관절 정보를 DataFrame에 추가
        row = [frame_number]
        row.extend(keypoints_info.tolist())
        df.loc[len(df)] = row

    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, dsize=(640, 480))
    cv2.imshow('YOLO Pose Detection', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('='*20)
        print('KeyBoard Interrupt')
        print('='*20)
        break

    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# DataFrame 출력
print(df)
