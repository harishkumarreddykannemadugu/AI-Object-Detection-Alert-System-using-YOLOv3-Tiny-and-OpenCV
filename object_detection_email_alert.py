import cv2
import time
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
import os
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
with open("coco.names", "r") as f:
classes = [line.strip() for line in f.readlines()]
SENDER_EMAIL = "nitinkumarreddym@gmail.com"
RECEIVER_EMAIL = "reddykharishkumar316@gmail.com"
APP_PASSWORD = "phfj dxot xxvk rlzc "
def send_email(video_path, label):
msg = EmailMessage()
msg["Subject"] = f"🚨 YOLO Alert: Detected {label}"
msg["From"] = SENDER_EMAIL
msg["To"] = RECEIVER_EMAIL
msg.set_content(f"An object was detected: {label}\nPlease find the attached video clip.")
with open(video_path, "rb") as f:
msg.add_attachment(f.read(), maintype="video", subtype="mp4", filename=os.path.basename(video_path))
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
server.login(SENDER_EMAIL, APP_PASSWORD)
server.send_message(msg)
print("✅ Email sent with video and object name.")
cap = cv2.VideoCapture(0)
alert_sent = False
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
while True:
ret, frame = cap.read()
if not ret:
break
height, width = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
boxes, confidences, class_ids = [], [], []
for out in outputs:
for detection in out:
scores = detection[5:]
class_id = int(scores.argmax())
confidence = scores[class_id]
if confidence > 0.5:
center_x = int(detection[0] * width)
center_y = int(detection[1] * height)
w = int(detection[2] * width)
h = int(detection[3] * height)
x = int(center_x - w / 2)
y = int(center_y - h / 2)
boxes.append([x, y, w, h])
confidences.append(float(confidence))
class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indexes) > 0 and not alert_sent:
detected_label = classes[class_ids[indexes[0][0]]]
print(f"[INFO] Detected: {detected_label}. Recording video...")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("detections", exist_ok=True)
video_path = f"detections/{detected_label}_{timestamp}.mp4"
out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
start_time = time.time()
while time.time() - start_time < 5:
ret, frame = cap.read()
if not ret:
break
out.write(frame)
cv2.imshow("Recording...", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
out.release()
send_email(video_path, detected_label)
alert_sent = True
cv2.imshow("YOLO Object Detection", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv2.destroyAllWindows()
