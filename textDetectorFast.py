import cv2
import pytesseract
from pytesseract import Output
from classes import Pokemon, Team
import json
import numpy as np
import time
import subprocess
import multiprocessing

start_time =0

def init_dev_mode(frames, roi_images, ocr_results):
    # Step through images manually
    current_index = 0
    total_frames = len(frames)

    while True:
        # Get current original frame and processed ROIs
        original_frame = frames[current_index]
        roi_overlays = roi_images[current_index]
        ocr_predictions = ocr_results[current_index]

        # Overlay processed ROIs on the original frame
        overlay_frame = original_frame.copy()

        for (processed_roi, (x, y, w, h)), (ocr_word, _) in zip(roi_overlays, ocr_predictions):
            # Convert processed ROI to 3-channel for visualization
            roi_bgr = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)
            roi_resized = cv2.resize(roi_bgr, (w, h))  # Resize back to original ROI size
            overlay_frame[y:y+h, x:x+w] = roi_resized  # Place processed ROI back onto the image

            # Draw detected word slightly above the bounding box
            if ocr_word:
                text_position = (x, y - 10)  # Slightly above the box
                cv2.putText(overlay_frame, ocr_word, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame with overlaid processed ROIs and text predictions
        cv2.imshow("OCR Frame Viewer (Left/Right to navigate)", overlay_frame)

        # Wait for user input
        key = cv2.waitKey(0)  # Wait indefinitely for a key press

        if key == 27:  # ESC key to exit
            break
        elif key == ord("a") or key == 81:  # Left arrow key ('a' for alternative)
            current_index = max(0, current_index - 1)  # Go back to the previous processed frame
        elif key == ord("d") or key == 83:  # Right arrow key ('d' for alternative)
            current_index = min(total_frames - 1, current_index + 1)  # Go forward to the next processed frame

    cv2.destroyAllWindows()  # Close all windows after exiting

def image_to_word_short(processed_roi):
        # Perform OCR using image_to_string (faster than image_to_data)
        best_word = pytesseract.image_to_string(processed_roi, config="--psm 6").strip()
        return best_word

def image_to_word_long(processed_roi):
    # Perform OCR on the processed ROI
    ocr_data = pytesseract.image_to_data(processed_roi, output_type=Output.DICT)

    best_word = ""
    highest_confidence = 0
    confidence_threshold = 40

    # Loop through detected words and find the highest-confidence one
    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])  # Confidence score (0-100)

        if word and confidence > confidence_threshold and confidence > highest_confidence:
            best_word = word
            highest_confidence = confidence

    return best_word

def preprocess_roi(roi):
    """Apply preprocessing techniques to suppress background noise while keeping text clear"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) # Upscale image for better OCR recognition
    gray = cv2.bilateralFilter(gray, d=15, sigmaColor=75, sigmaSpace=75)     # Apply a strong bilateral filter to smooth out background while preserving edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)     # Apply a larger Gaussian blur to reduce small noise elements
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) # Morphological Opening: Removes small noise while keeping text intact
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8)) # Morphological Closing: Enhances the structure of text while suppressing noise
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Adaptive thresholding - dynamically adjusts for lighting inconsistencies
                                 cv2.THRESH_BINARY, 11, 2)
    return gray

import subprocess
import datetime

def get_video_metadata(video_path):
    """Returns width, height, total frames, duration, and recording time using FFmpeg."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,duration",
        "-show_entries", "format_tags=creation_time",
        "-of", "csv=p=0",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = result.stdout.strip().split("\n")
        
        # Debugging: Print metadata to check what is actually returned
        print(f"🔍 Debug: Metadata returned by ffprobe → {metadata}")

        # Ensure metadata list has at least 2 elements
        width, height = map(int, metadata[0].split(",")[:2]) if len(metadata) > 0 else ("Unknown", "Unknown")
        total_frames = int(metadata[1]) if len(metadata) > 1 and metadata[1].isdigit() else "Unknown"
        duration = float(metadata[2]) if len(metadata) > 2 and metadata[2].replace(".", "", 1).isdigit() else "Unknown"
        recorded_time = metadata[3] if len(metadata) > 3 else "Unknown"

        # Convert recorded_time to datetime format if valid
        if recorded_time != "Unknown":
            try:
                recorded_time = datetime.datetime.fromisoformat(recorded_time.replace("Z", "+00:00"))
            except ValueError:
                pass  # Keep it as a string if formatting fails

        return width, height, total_frames, duration, recorded_time

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ffprobe: {e}")
        return "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"

def crop_video_to_memory(input_video, roi, video_width, video_height, frame_interval=30):
    """Uses FFmpeg to crop a video, convert to grayscale, and store every 30th frame in memory."""
    x = int(roi[0] * video_width)
    y = int(roi[1] * video_height)
    w = int(roi[2] * video_width)
    h = int(roi[3] * video_height)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_video,             # Input file
        "-vf", f"crop={w}:{h}:{x}:{y}", # Crop filter
        "-f", "rawvideo",               # Output as raw frames
        "-pix_fmt", "gray",             # Convert to grayscale to save space
        "-"                             # Output to stdout (pipe)
    ]

    # Run FFmpeg and capture raw frame data
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_size = w * h  # 1 byte per pixel (Grayscale)
    frames = []
    frame_count = 0


    while True:
        raw_frame = process.stdout.read(frame_size)  # Read frame from stdout
        if not raw_frame:
            break  # Stop when no more frames are available

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  # Skip frames that are not every 30th frame

        frame = np.frombuffer(raw_frame, np.uint8).reshape((h, w))  # Convert raw data to NumPy array
        frames.append(frame)  # Store frame in memory

    process.stdout.close()
    process.wait()

    print(f"✅ Stored {len(frames)} cropped grayscale frames in memory.")
    return frames  # Returns frames as NumPy arrays

def process_video(file_path, frame_interval, dev_mode):
    # Start timer for debugging
    start_time = time.time()

    # Get video width and height
    video_width, video_height, total_frames, video_duration, video_recorded_time = get_video_metadata(file_path)

    # Define multiple Regions of Interest (ROIs) as fractions of the video dimensions
    screen_x = 1179
    screen_y = 2556
    ROIs = [
        ((30/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "Self"),  # My Pokemon
        ((860/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "Opponent"),  # Opponent Pokemon
    ]

    my_team = Team()
    opponent_team = Team()

    for roi in ROIs:
        cropped_video =  crop_video_to_memory(file_path, roi, video_width, video_height, frame_interval=30)
        for frame in cropped_video:
            best_word = image_to_word_long(frame)

            if best_word:
                if roi[4] == "Opponent":
                    if not opponent_team.has_pokemon(best_word):
                        new_pokemon = Pokemon(best_word)
                        opponent_team.add_pokemon(new_pokemon)
                elif roi[4] == "Self":
                    if not my_team.has_pokemon(best_word):
                        new_pokemon = Pokemon(best_word)
                        my_team.add_pokemon(new_pokemon)
    
    print("My Team: ")
    my_team.print_team()
    print("Opponent Team: ")
    opponent_team.print_team()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n🚀 Execution Time: {execution_time:.2f} seconds")

def main():
    file_path = "src/video.mp4";
    frame_interval = 30;
    dev_mode = False;
    process_video(file_path, frame_interval, dev_mode)

if __name__ == "__main__":
    main()