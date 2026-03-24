import cv2
import mediapipe as mp
import csv

# --- Settings ---
# Enter the path to your video
INPUT_VIDEO = 'path/to/your/video_cam1.mp4' 
# Enter the name of the file to save the results
OUTPUT_CSV = 'coords_cam1.csv' 
# Whether to display the video during processing (significantly slows down processing)
SHOW_VIDEO = True 

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Configuration: static_image_mode=False (for video), high confidence
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,       # 0, 1, or 2 (higher is more accurate but slower)
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- List of keypoint names (for the CSV header) ---
# MediaPipe outputs 33 points. We will create a header: point0_x, point0_y, point1_x...
landmark_names = [f"point{i}_{coord}" for i in range(33) for coord in ['x', 'y']]
csv_header = ['frame_number'] + landmark_names

def process_video():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}")
        return

    # Get video parameters for logging
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {INPUT_VIDEO}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}")

    # Create/open CSV file for writing
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header) # Write the header

        current_frame = 0
        while cap.isOpened():
            success, frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Read a frame. Skip if the video has ended
            success, frame = cap.read()
            if not success:
                break

            current_frame += 1
            if current_frame % 50 == 0:
                print(f"Processed frame {current_frame}/{frame_count}")

            # 1. Convert BGR to RGB (MediaPipe requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Process the frame with MediaPipe
            results = pose.process(rgb_frame)

            # 3. Extract coordinates
            frame_row = [current_frame] # Start the data row with the frame number

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    # MediaPipe outputs normalized coordinates (0.0 - 1.0)
                    # Convert them into pixel coordinates of the image
                    # landmark.z is the relative depth (not used for now)
                    pixel_x = lm.x * w
                    pixel_y = lm.y * h
                    frame_row.extend([pixel_x, pixel_y])
                
                # If SHOW_VIDEO=True, draw the landmarks on the frame
                if SHOW_VIDEO:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                # If no person is detected, fill the row with zeros (or None)
                frame_row.extend([0.0] * (33 * 2))

            # 4. Write the data row to the CSV
            writer.writerow(frame_row)

            # Display the frame (if enabled)
            if SHOW_VIDEO:
                cv2.imshow('MediaPipe Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    print(f"Done. Coordinates saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_video()