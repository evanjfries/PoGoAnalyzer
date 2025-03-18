import cv2

def step_through_video(video_path, step=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Reached end of video or failed to read frame.")
            break

        # Display frame number (frame_index // 30)
        display_text = f"Frame: {frame_index // step}"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Video Stepper", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('d') or key == 83:  # Right arrow or 'd'
            frame_index = min(frame_index + step, total_frames - 1)
        elif key == ord('a') or key == 81:  # Left arrow or 'a'
            frame_index = max(frame_index - step, 0)
        elif key == ord('q'):  # 'q'
            frame_index = max(frame_index - (step* 10), 0)
        elif key == ord('e') or key == 81:  # 'e'
            frame_index = min(frame_index + (step * 10), total_frames - 1)
        elif key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
step_through_video("src/10min.mp4", step=30)
