import cv2
import numpy as np
import time

# Define constants
TRAFFIC_LIGHT_WIDTH, TRAFFIC_LIGHT_HEIGHT = 200, 500
COUNT_LINE_POSITION = 550
MIN_WIDTH_REACT, MIN_HEIGHT_REACT = 80, 80

# Signal timing limits
MAX_GREEN_TIME = 60
MIN_GREEN_TIME = 10
YELLOW_TIMING = 5
RED_TIME = 30

# Video file names for four lanes
video_files = ['EMPTYROAD.webm', 'PROJ.mp4', 'EMPTYROAD.webm', 'PROJ.mp4']

# Initialize video capture for four lanes
caps = [cv2.VideoCapture(video) for video in video_files]

# Check if videos are loaded correctly
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Failed to open video for lane {i+1}.")

# Load YOLOv3
net = cv2.dnn.readNet("yolov3 (1).weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels for detection
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Vehicle counting variables for each lane
detect_lanes = [[], [], [], []]  # For four lanes
counters = [0, 0, 0, 0]  # Vehicle counters for each lane
offset = 6

# Traffic light states and timing control for each lane
signal_states = ["green", "red", "red", "red"]  # Initial states, only lane 1 is green
last_switch_times = [time.time() for _ in range(4)]  # Correct initialization

# Function to determine the center of a bounding box
def center_handle(x, y, w, h):
    return x + w // 2, y + h // 2

# Function to dynamically update green time based on vehicle count
def update_green_time(vehicle_count):
    return min(MAX_GREEN_TIME, max(MIN_GREEN_TIME, 10 + vehicle_count * 2))

# Create an empty traffic light image
traffic_light = np.zeros((TRAFFIC_LIGHT_HEIGHT, TRAFFIC_LIGHT_WIDTH, 3), dtype=np.uint8)

# Function to restart a video if it ends
def restart_video(cap, video_file):
    cap.release()
    return cv2.VideoCapture(video_file)

# Main processing loop
while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from video for lane {i+1}. Restarting video.")
            caps[i] = restart_video(caps[i], video_files[i])
            ret, frame = caps[i].read()
            if not ret:
                print(f"Failed to load frame from video for lane {i+1}. Continuing.")
                continue
        frames.append(frame)

    for lane_index in range(4):  # Use lane_index to iterate over lanes
        # Prepare the frame for YOLOv3
        blob = cv2.dnn.blobFromImage(frames[lane_index], 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Detect objects in the frame
        height, width, channels = frames[lane_index].shape
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 2:  # Class 2 is car/bike/bus in COCO dataset
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for index in range(len(boxes)):  # Use index as the iterator variable
            if index in indexes.flatten():  # Properly check the indexes from NMS
                x, y, w, h = boxes[index]
                center = center_handle(x, y, w, h)
                detect_lanes[lane_index].append(center)  # Keep adding the vehicle centers
                cv2.rectangle(frames[lane_index], (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frames[lane_index], center, 4, (0, 0, 255), -1)

        detect_lanes[lane_index] = [d for d in detect_lanes[lane_index] if COUNT_LINE_POSITION - offset < d[1] < COUNT_LINE_POSITION + offset]
        counters[lane_index] += len(detect_lanes[lane_index])

        # Signal timing logic
        current_time = time.time()
        elapsed_time = current_time - last_switch_times[lane_index]
        green_time = update_green_time(counters[lane_index])

        if signal_states[lane_index] == "green" and elapsed_time >= green_time:
            signal_states[lane_index] = "yellow"
            last_switch_times[lane_index] = current_time
        elif signal_states[lane_index] == "yellow" and elapsed_time >= YELLOW_TIMING:
            signal_states[lane_index] = "red"
            last_switch_times[lane_index] = current_time
            next_lane = (lane_index + 1) % 4
            signal_states[next_lane] = "green"
            last_switch_times[next_lane] = current_time

        cv2.putText(frames[lane_index], f"Lane {lane_index+1} Count: {counters[lane_index]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display all windows
    for lane_index in range(4):
        lane_light = traffic_light.copy()
        if signal_states[lane_index] == "green":
            cv2.circle(lane_light, (100, 150), 50, (0, 255, 0), -1)
        elif signal_states[lane_index] == "yellow":
            cv2.circle(lane_light, (100, 300), 50, (0, 255, 255), -1)
        else:
            cv2.circle(lane_light, (100, 450), 50, (0, 0, 255), -1)

        cv2.putText(lane_light, f"Lane {lane_index+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(lane_light, f"Green Time: {update_green_time(counters[lane_index])}s", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(lane_light, f"Yellow Time: {YELLOW_TIMING}s", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(lane_light, f"Red Time: {RED_TIME}s", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show each lane's traffic light in different window
        cv2.imshow(f'Traffic Light - Lane {lane_index+1}', lane_light)
        cv2.imshow(f'Vehicle Detection - Lane {lane_index+1}', frames[lane_index])

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
