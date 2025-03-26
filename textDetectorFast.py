import cv2
import pytesseract
from pytesseract import Output
from classes import Pokemon, Team, Battle
import json
import numpy as np
import time
import subprocess
import multiprocessing
import os
import datetime

start_time =0

def merge_battle_breaks(team1, team2):
    # Combine both lists and sort by start time
    combined = sorted(team1 + team2)
    
    # Merge overlapping and adjacent intervals
    merged = []
    for start, end in combined:
        if merged and merged[-1][1] >= start - 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged

def image_to_word_long(processed_roi):
    # Perform OCR on the processed ROI
    ocr_data = pytesseract.image_to_data(processed_roi, output_type=Output.DICT)

    label = ""
    highest_confidence = 0
    confidence_threshold = 40

    # Loop through detected words and find the highest-confidence one
    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])  # Confidence score (0-100)

        if word and confidence > confidence_threshold and confidence > highest_confidence:
            label = word
            highest_confidence = confidence

    return label

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

def get_video_metadata(video_path):
    """Returns width, height, total_frames (calculated), duration, FPS, and recording time using FFmpeg."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format_tags=creation_time",
        "-of", "csv=p=0",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = result.stdout.strip().split("\n")

        print(f"🔍 Debug: Metadata returned by ffprobe → {metadata}")

        # Parse first line for width, height, fps, duration
        stream_info = metadata[0].split(",")
        width = int(stream_info[0])
        height = int(stream_info[1])

        fps_raw = stream_info[2]
        if "/" in fps_raw:
            num, denom = map(int, fps_raw.split("/"))
            fps = num / denom if denom != 0 else 0
        else:
            fps = float(fps_raw)

        duration = float(stream_info[3])
        total_frames = int(fps * duration)

        # Parse recording time if present
        recorded_time = metadata[1] if len(metadata) > 1 else "Unknown"
        if recorded_time != "Unknown":
            try:
                recorded_time = datetime.datetime.fromisoformat(recorded_time.replace("Z", "+00:00"))
            except ValueError:
                pass  # Leave as string if parsing fails

        return width, height, total_frames, duration, fps, recorded_time

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ffprobe: {e}")
        return "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"

def euclidean_distance(frame1, frame2):
    """Computes the Euclidean distance between two frames."""
    return np.linalg.norm(frame1.astype(np.float32) - frame2.astype(np.float32))

def save_npy(file, file_name, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    npy_path = os.path.join(dir_name, f"{file_name}.npy")
    np.save(npy_path, file)

def save_img(file, file_name, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    img_path = os.path.join(dir_name, f"{file_name}.png")
    cv2.imwrite(img_path, file)

def calculate_roi(roi, video_width, video_height):
    x = int(roi[0] * video_width)
    y = int(roi[1] * video_height)
    w = int(roi[2] * video_width)
    h = int(roi[3] * video_height)
    return x,y,w,h

def word_in_file(word, file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return word in (line.strip() for line in file)

def determine_match_outcomes(input_video, roi, video_width, video_height, frame_ranges):
    x,y,w,h = calculate_roi(roi, video_width, video_height)

    start_frame_offset=0
    if roi[4] == "Win":
        # start_frame_offset = 15 # Saves extra time, as the match result never appears within 15 frames battle end
        reference_frame_path = "src/references/Win.npy"
    elif roi[4] == "Loss":
        # start_frame_offset = 7
        reference_frame_path = "src/references/Loss.npy"
    max_distance_from_reference = 10000
    
    reference_frame = None
    if os.path.exists(reference_frame_path):
        reference_frame = np.load(reference_frame_path)
        reference_frame = cv2.resize(reference_frame, (w, h))  # Resize reference to match

    results = []
    for start_frame, end_frame in frame_ranges:
        start_frame += start_frame_offset
        frame_count_to_read = end_frame - start_frame + 1
        start_time_seconds = start_frame / 2 # Based on 2 fps

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", str(start_time_seconds),       # Seek to start time
            "-i", input_video,
            "-vf", f"scale={video_width}:{video_height},fps=2,crop={w}:{h}:{x}:{y},format=gray",
            "-frames:v", str(frame_count_to_read),  # Process only this many frames
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-"
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

        frame_size = w * h
        frame_number = start_frame

        while True:
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((h, w))
            distance = euclidean_distance(frame, reference_frame)

            if distance < max_distance_from_reference:
                results.append(frame_number)
                break

            frame_number += 1

        process.stdout.close()
        process.wait()

    print(f"✅ {roi[4]} → Found {len(results)}.")
    return roi[4], results

def crop_video_to_memory(input_video, roi, video_width, video_height):
    x,y,w,h = calculate_roi(roi, video_width, video_height)

    if roi[4] == "My Team":
        reference_frame_path = "src/references/MyTeam.npy"
        max_distance_from_reference = 3000
        min_distance_from_previous = 1000
    elif roi[4] == "Opponent Team":
        reference_frame_path = "src/references/OpponentTeam.npy"
        max_distance_from_reference = 3000
        min_distance_from_previous = 1000

    if os.path.exists(reference_frame_path):
        reference_frame_npy = np.load(reference_frame_path)
        reference_frame_npy = cv2.resize(reference_frame_npy, (w, h))  # Ensure same dimensions

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"scale={video_width}:{video_height},fps=2,crop={w}:{h}:{x}:{y},format=gray",  # Explicit scaling
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]

    # Run FFmpeg and capture raw frame data
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    
    frame_size = w * h 
    pokemon = []
    frame_number = 0
    prev_frame = None
    battle_breaks = []

    while True:
        raw_frame = process.stdout.read(frame_size)  # Read frame from stdout
        if not raw_frame:
            battle_breaks.append((prev_frame[1], frame_number)) # Add the last frame with a pokemon
            break  # Stop when no more frames are available

        frame_npy = np.frombuffer(raw_frame, np.uint8).reshape((h, w)) # Convert raw data to NumPy array
        frame = (frame_npy, frame_number)  

        distance = euclidean_distance(frame[0], reference_frame_npy)

        if distance < max_distance_from_reference: # Continue if it contains the data we need

            # save_npy(frame[0], f"{frame_number}-{distance:.0f}", f"src/{roi[4]}-npy")

            if prev_frame is not None: # If we are not on the first frame

                # Discard if too similar to last frame (repeat data)
                distance_from_prev = euclidean_distance(frame[0], prev_frame[0])
                # save_img(frame[0], f"{frame_number}-{distance:.0f}-{distance_from_prev:.0f}", f"src/{roi[4]}-images")
                
                # Add as a frame of interest if last frame was over 30 ago
                time_since_last_pokemon = frame[1] - prev_frame[1]
                if time_since_last_pokemon > 30:
                    battle_breaks.append((prev_frame[1], frame[1]))

                # If too close to previous --> same pokemon as before
                if distance_from_prev < min_distance_from_previous:
                    frame_number+=1
                    prev_frame = frame
                    continue  

            else: # We are on the first battle frame; add it as a frame of interest
                battle_breaks.append((0, frame_number))

            label = process_frame_ocr(frame[0])
            if word_in_file(label,"src/references/pokemon.txt") and label is not None:
                # print(f"{label}")
                prev_frame = frame
                pokemon.append((label, frame_number))
            # save_npy(frame, f"frame_{saved_count:04d}", "src/npy")

        frame_number += 1

    process.stdout.close()
    process.wait()

    print(f"✅ {roi[4]} → Stored {len(pokemon)} cropped grayscale frames in memory.")
    return roi[4], pokemon, battle_breaks  # Returns frames as NumPy arrays

def process_frame_ocr(frame):
    label = image_to_word_long(frame)
    return label if label else None

def process_video(file_path, frame_interval, dev_mode):
    # Start timer for debugging
    start_time = time.time()

    # Get video width and height
    video_width, video_height, total_frames, video_duration, fps, video_recorded_time = get_video_metadata(file_path)
    # fps = int(video_duration)/int(total_frames)
    # print(f"FPS: {fps}")
    # print(f"{video_duration}")
    # print(f"{fps}")

    # Define multiple Regions of Interest (text_ROIs) as fractions of the video dimensions
    screen_x = 1179
    screen_y = 2556
    text_ROIs = [
        ((30/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "My Team"),  # My Pokemon
        ((860/screen_x), (170/screen_y), (280/screen_x), (40/screen_y), "Opponent Team"),  # Opponent Pokemon
    ]

    image_ROIs = [
        ((180/screen_x), (1215/screen_y), (805/screen_x), (135/screen_y), "Win"),  # Win
        ((330/screen_x), (1110/screen_y), (515/screen_x), (135/screen_y), "Loss"),  # Loss
    ]

    pokemon_results = []
    battle_results = []
    # Step 1: Crop all text ROIs
    with multiprocessing.Pool(processes=len(text_ROIs)) as pool:
        pokemon_results = pool.starmap(
            crop_video_to_memory,
            [(file_path, roi, video_width, video_height) for roi in text_ROIs]
        )

    # Step 2: Determine battle outcomes
    battle_breaks = [battle_break for _, _, battle_break in pokemon_results]
    merged_battle_breaks = merge_battle_breaks(battle_breaks[0], battle_breaks[1])

    # print(f"Battle Breaks (My Team): {battle_breaks[0]}")
    # print(f"\nBattle Breaks (Opponent Team): {battle_breaks[1]}")
    # print(f"\nMerged Battle Breaks: {merged_battle_breaks}")
    with multiprocessing.Pool(processes=len(image_ROIs)) as battle_pool:
        battle_results = battle_pool.starmap(
            determine_match_outcomes,
            [(file_path, roi, video_width, video_height, merged_battle_breaks) for roi in image_ROIs]
        )

    results_array = []    
    for result in battle_results:
        if result:
            roi_label, results = result
            # print(f"{roi_label} : {results}")
            for result in results:
                results_array.append((roi_label, result))
    
    results_array.sort(key=lambda x: x[1])
    # Sort pokemon_results by frame_number just in case
    # pokemon_results = [r for r in pokemon_results if r is not None]
    # Now safe to sort
    # pokemon_results.sort(key=lambda r: r[2])
    battles = []
    for result in pokemon_results:
        if result[0] == "Opponent Team":
            opponent_team = result
        elif result[0] == "My Team":
            my_team = result

    # For each battle range
    for i, (battle_result_label, battle_end_frame) in enumerate(results_array):
        # Determine start frame for this battle
        battle_start_frame = 0 if i == 0 else results_array[i-1][1] + 1

        my_team = Team("My Team")
        opponent_team = Team("Opponent Team")

        # Filter Pokémon results within this battle's frame range
        for result in pokemon_results:
            # print(result)
            roi_label, labels, battle_breaks = result  # Unpack the single tuple

            for label in labels:  # Now loop through the Pokémon list
                pokemon = label[0]
                frame_number = label[1]

                if battle_start_frame <= frame_number <= battle_end_frame:
                    if roi_label == "Opponent Team" and not opponent_team.has_pokemon(pokemon):
                        opponent_team.add_pokemon(Pokemon(pokemon))
                    elif roi_label == "My Team" and not my_team.has_pokemon(pokemon):
                        my_team.add_pokemon(Pokemon(pokemon))

        battle = Battle(my_team, opponent_team, battle_result_label)
        battles.append(battle)

    i = 1
    for battle in battles:
        print(f"\n🛡️ Battle {i}")
        battle.print_battle()
        i+=1
    
    win_count = sum(battle.is_win() for battle in battles)
    loss_count = len(battles) - win_count
    print(f"\n🏆 Wins: {win_count}")
    print(f"💀 Losses: {loss_count}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n🚀 Execution Time: {execution_time:.2f} seconds")

def main():
    file_path = "src/hour.mp4";
    frame_interval = 30;
    dev_mode = False;
    process_video(file_path, frame_interval, dev_mode)

if __name__ == "__main__":
    main()