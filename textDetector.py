import cv2
import pytesseract
from pytesseract import Output
import json
import numpy as np
import time

start_time =0

class Pokemon:
    def __init__(self, name):
        self.name = name
        self.moves = []

    def get_name(self):
        return self.name
    
    def add_move(self, move): 
        self.moves.append(move)
    
    def get_move_count(self):
        return len(self.moves)
        

class Team:
    def __init__(self):
        self.pokemon = []
    
    def add_pokemon(self, pokemon):
        self.pokemon.append(pokemon)
    
    def get_pokemon_count(self):
        return len(self.pokemon)
    
    def has_pokemon(self, pokemon_name):
        return any(p.get_name() == pokemon_name for p in self.pokemon)
    
    def print_team(self):
        members = []
        for pokemon in self.pokemon:
            members.append(pokemon.get_name())
        print(members)

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

def process_video(file_path, frame_interval, dev_mode):
    # Start timer for debugging
    start_time = time.time()
    
    # Load video
    cap = cv2.VideoCapture(file_path)

    # Get video width and height
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0

    # Define multiple Regions of Interest (ROIs) as fractions of the video dimensions
    screen_x = 1179
    screen_y = 2556
    ROIs = [
        ((30/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "Self"),  # My Pokemon
        ((860/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "Opponent"),  # Opponent Pokemon
    ]

    my_team = Team()
    opponent_team = Team()

    # Dev mode storage:
    if dev_mode:
        frames = []  # Store original frames
        roi_images = []  # Store processed ROIs for overlay
        ocr_results = []  # Store OCR results

    # Process frames at specific intervals without reading the full video
    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Jump to the desired frame
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # print(f"Processing frame {frame_count}")  # Debugging

        # Prepare the display for dev mode
        if dev_mode:
            frames.append(frame.copy())  # Store the original frame

        roi_processed = []
        frame_ocr_results = []

        for roi in ROIs:
            x = int(roi[0] * video_width)
            y = int(roi[1] * video_height)
            w = int(roi[2] * video_width)
            h = int(roi[3] * video_height)

            roi_frame = frame[y:y+h, x:x+w]  # Extract only the ROI
            processed_roi = preprocess_roi(roi_frame)  # Apply OCR preprocessing
            best_word = image_to_word_long(processed_roi)  # Perform OCR
            # best_word = "Bob"

            if best_word:
                if roi[4] == "Opponent":
                    if not opponent_team.has_pokemon(best_word):
                        new_pokemon = Pokemon(best_word)
                        opponent_team.add_pokemon(new_pokemon)
                elif roi[4] == "Self":
                    if not my_team.has_pokemon(best_word):
                        new_pokemon = Pokemon(best_word)
                        my_team.add_pokemon(new_pokemon)

            if dev_mode:
                roi_processed.append((processed_roi, (x, y, w, h)))  # Store processed ROI with its position
                frame_ocr_results.append((best_word, (x, y, w, h)))  # Store OCR results with position

        if dev_mode:
            roi_images.append(roi_processed)  # Store processed ROIs per frame
            ocr_results.append(frame_ocr_results)  # Store OCR predictions per frame

        # Move to the next frame interval
        frame_count += frame_interval

    cap.release()  # Release video since we've stored all frames

    print("My Team: ")
    my_team.print_team()
    print("Opponent Team: ")
    opponent_team.print_team()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n🚀 Execution Time: {execution_time:.2f} seconds")

    if dev_mode:
        init_dev_mode(frames, roi_images, ocr_results)


def main():
    file_path = "src/video.mp4";
    frame_interval = 30;
    dev_mode = False;
    process_video(file_path, frame_interval, dev_mode)

if __name__ == "__main__":
    main()