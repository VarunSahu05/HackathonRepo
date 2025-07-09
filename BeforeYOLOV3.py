import cv2
import numpy as np
import time

# Constants for traffic light dimensions
TRAFFIC_LIGHT_WIDTH, TRAFFIC_LIGHT_HEIGHT = 200, 500
COUNT_LINE_POSITION = 550
MIN_WIDTH_REACT, MIN_HEIGHT_REACT = 80, 80

# Traffic light timing constants
MAX_GREEN_TIME = 60
MIN_GREEN_TIME = 10
YELLOW_TIMING = 5
RED_TIME = 30

# Input video files for simulation
video_files = ['EMPTYROAD.webm', 'PROJ.mp4', 'EMPTYROAD.webm', 'PROJ.mp4']

# Initialize video capture and background subtraction algorithm
caps = [cv2.VideoCapture(video) for video in video_files]
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_files[i]}")
        exit()

algos = [cv2.createBackgroundSubtractorMOG2() for _ in range(4)]

# Lists to store vehicle data for each lane
detect_lanes = [set() for _ in range(4)]  # Use sets to avoid duplicate entries
counters = [0, 0, 0, 0]
offset = 6  # Range for counting vehicles crossing the line

# Initial signal states and timing for each lane
signal_states = ["green", "red", "red", "red"]
last_switch_times = [time.time() for _ in range(4)]

def center_handle(x, y, w, h):
    """Calculate the center of a bounding box."""
    return x + w // 2, y + h // 2

def update_green_time(vehicle_count):
    """Update the green light duration based on vehicle count."""
    return min(MAX_GREEN_TIME, max(MIN_GREEN_TIME, 10 + vehicle_count * 2))

# Placeholder for traffic light images
traffic_light = np.zeros((TRAFFIC_LIGHT_HEIGHT, TRAFFIC_LIGHT_WIDTH, 3), dtype=np.uint8)

def restart_video(cap, video_file):
    """Restart the video if it's finished or error occurs."""
    cap.release()
    new_cap = cv2.VideoCapture(video_file)
    if not new_cap.isOpened():
        print(f"Error: Unable to open video {video_file}")
    return new_cap

def draw_traffic_light(lane_light, lane, signal_state, vehicle_count):
    """Helper function to display traffic light signals and timings."""
    if signal_state == "green":
        cv2.circle(lane_light, (100, 150), 50, (0, 255, 0), -1)  # Green light
    elif signal_state == "yellow":
        cv2.circle(lane_light, (100, 300), 50, (0, 255, 255), -1)  # Yellow light
    else:
        cv2.circle(lane_light, (100, 450), 50, (0, 0, 255), -1)  # Red light

    cv2.putText(lane_light, f"Lane {lane+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(lane_light, f"Green Time: {update_green_time(vehicle_count)}s", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(lane_light, f"Yellow Time: {YELLOW_TIMING}s", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(lane_light, f"Red Time: {RED_TIME}s", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Main loop for processing video frames and controlling traffic lights
while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: Failed to read frame from lane {i+1}. Restarting video...")
            caps[i] = restart_video(caps[i], video_files[i])
            ret, frame = caps[i].read()
            if not ret or frame is None:
                print(f"Error: Unable to read frame from lane {i+1} even after restart.")
                continue
        frames.append(frame)

    for i in range(4):
        if frames[i] is not None:
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            blur = cv2.GaussianBlur(gray, (3, 3), 5)  # Apply Gaussian blur
            mask = algos[i].apply(blur)  # Apply background subtraction
            dilated = cv2.dilate(mask, np.ones((5, 5)))  # Dilation to enhance the mask
            dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5)))  # Morphological closing

            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
            cv2.line(frames[i], (25, COUNT_LINE_POSITION), (frames[i].shape[1] - 25, COUNT_LINE_POSITION), (255, 127, 0), 2)  # Line for counting vehicles

            current_detected = set()  # To track vehicles detected in this frame
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)  # Get bounding box for each contour
                if w >= MIN_WIDTH_REACT and h >= MIN_HEIGHT_REACT:  # Only consider large contours
                    center = center_handle(x, y, w, h)
                    if COUNT_LINE_POSITION - offset < center[1] < COUNT_LINE_POSITION + offset:
                        current_detected.add(center)
                        cv2.rectangle(frames[i], (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
                        cv2.circle(frames[i], center, 4, (0, 0, 255), -1)  # Mark the center of the vehicle

            # Update counter for new vehicles that crossed the line
            new_vehicles = current_detected - detect_lanes[i]  # Vehicles not seen in previous frame
            counters[i] += len(new_vehicles)
            detect_lanes[i] = current_detected  # Update detected vehicles for this frame

            current_time = time.time()
            elapsed_time = current_time - last_switch_times[i]
            green_time = update_green_time(counters[i])  # Calculate green time for the lane

            # Signal state transition logic
            if signal_states[i] == "green" and elapsed_time >= green_time:
                signal_states[i] = "yellow"
                last_switch_times[i] = current_time
            elif signal_states[i] == "yellow" and elapsed_time >= YELLOW_TIMING:
                signal_states[i] = "red"
                last_switch_times[i] = current_time
                next_lane = (i + 1) % 4  # Move to the next lane
                signal_states[next_lane] = "green"
                last_switch_times[next_lane] = current_time

            # Display vehicle count on the frame
            cv2.putText(frames[i], f"Lane {i+1} Count: {counters[i]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for i in range(4):
        lane_light = traffic_light.copy()
        draw_traffic_light(lane_light, i, signal_states[i], counters[i])

        # Display the traffic light and vehicle detection frame
        cv2.imshow(f'Traffic Light - Lane {i+1}', lane_light)
        cv2.imshow(f'Vehicle Detection - Lane {i+1}', frames[i])

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and close windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
