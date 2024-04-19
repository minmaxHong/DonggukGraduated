import torch
import cv2
from ultralytics import YOLO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIDEO_PATH = r"C:\Users\H_\Desktop\Sungmin_Github\DonggukGraduated\videoplayback.mp4"
OUTPUT_PATH = r"C:\Users\H_\Desktop\Sungmin_Github\DonggukGraduated\output_video.mp4"

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('='*20)
        print('No Frame')
        print('='*20)
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()  
    
    cv2.imshow('YOLO Pose Detection', annotated_frame)
    out.write(annotated_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print('='*20)
        print('KeyBoard Interrupt')
        print('='*20)
        break


cap.release()
out.release()
cv2.destroyAllWindows()
