import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path
import time
import threading

def play_video_in_plot_controlled_speed(video_path,
                                        plot_interval_ms=33,
                                        target_video_fps=30.0,
                                        target_width=None):
    """
    Plays a video file in a Matplotlib window with independent control over
    video playback speed and plot update rate.

    Args:
        video_path (str or Path): Path to the video file.
        plot_interval_ms (int): Delay between plot update calls (ms).
                                Controls the responsiveness of the plot itself.
        target_video_fps (float): Desired playback speed of the video in FPS.
        target_width (int, optional): Resize frames to this width. Defaults to None.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("Error: Could not get frame count or video is empty.")
        cap.release()
        return
    print(f"Video Info: {width}x{height}, Original FPS: {original_fps:.2f}, Frames: {frame_count}")
    print(f"Plot Update Interval: {plot_interval_ms}ms (~{1000/plot_interval_ms:.1f} Hz)")
    print(f"Target Video Playback Speed: {target_video_fps:.1f} FPS")


    # --- Resize Helper ---
    def _resize(frame, tw):
        if tw is None or frame is None or width == 0: return frame
        if width == tw: return frame
        ratio = height / width
        nw = int(tw)
        nh = int(nw * ratio)
        if nw <= 0 or nh <= 0: return frame
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    # --- Read the first frame ---
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_display = _resize(frame_rgb, target_width)

    # --- Set up the plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Playing: {video_path.name} (Target Vid FPS: {target_video_fps:.1f})", fontsize=10)
    ax.axis('off')
    #im = ax.imshow(frame_display)
    im = ax.imshow(frame_rgb) # Display the first frame directly

    # --- State variables for playback control ---
    playback_start_time = time.monotonic() # Use monotonic clock for elapsed time
    last_displayed_frame_num = 0
    is_playing = True # Add a flag to potentially pause

    # --- Animation update function ---
    def update_frame(frame_idx):
        nonlocal playback_start_time, last_displayed_frame_num # Allow modification

        if not is_playing or not cap.isOpened():
            return im,

        # 1. Calculate which video frame *should* be displayed now
        current_time = time.monotonic()
        elapsed_time = time.monotonic() - playback_start_time
        target_frame_num_float = elapsed_time * target_video_fps

        # 2. Handle looping
        target_frame_num = int(round(target_frame_num_float)) % frame_count

        # 3. Optimization: Only seek and read if the target frame is different
        #    from the last one we attempted to display.
        #    (Note: cap.read() also advances the frame, so seeking is usually necessary)
        #if target_frame_num != last_displayed_frame_num: # Check may not be needed if always seeking

        # 4. Seek to the calculated frame position
        #    Use CAP_PROP_POS_FRAMES. Setting this property might be approximate.
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret_up, frame_up = cap.read() # Read the frame at (or near) the target pos

        if ret_up:
            # Process and display the frame
            frame_rgb_up = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)
            frame_display_up = _resize(frame_rgb_up, target_width)
            im.set_data(frame_display_up)
            last_displayed_frame_num = target_frame_num # Store the frame number we aimed for
        else:
            # Read failed even after setting position (might happen at loop boundary?)
            print(f"Warning: Failed to read frame {target_frame_num} after seeking.")
            # Keep showing the previous frame (im hasn't been updated)
            pass

        return im,

    # --- Create and run the animation ---
    ani = FuncAnimation(fig, update_frame, interval=plot_interval_ms, blit=False, cache_frame_data=False)


    try:
        plt.show()
    finally:
        print("\nPlot closed. Releasing video capture.")
        if cap.isOpened():
            cap.release()

def _resize(frame, width, height, tw):
    if tw is None or frame is None or width == 0: return frame
    if width == tw: return frame
    ratio = height / width
    nw = int(tw)
    nh = int(nw * ratio)
    if nw <= 0 or nh <= 0: return frame
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)



# --- Example Usage ---
if __name__ == "__main__":
    video_file = Path("./media/videos/IMG_4938.MOV") # YOUR VIDEO FILE
    frame = None
    lock = threading.Lock()

    def video_thread(path, skip_frames=1):
        global frame
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            for _ in range(skip_frames - 1):
                cap.read()  # Discard

            ret, f = cap.read()
            if not ret:
                break
            with lock:
                frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        cap.release()

    # Start video reader thread
    threading.Thread(target=video_thread, args=(video_file, 8), daemon=True).start()

    # Matplotlib setup
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # placeholder frame

    def update(_):
        global frame
        with lock:
            if frame is not None:
                im.set_data(frame)
        return [im]

    ani = FuncAnimation(fig, update, interval=5, blit=True)
    plt.show()

    # if not video_file.exists():
    #     print(f"ERROR: Video file not found at '{video_file}'")
    # else:
    #     # --- Experiment ---
    #     plot_update_rate_hz = 60 # How often to update the plot (e.g., 30Hz)
    #     plot_interval = int(round(1000 / plot_update_rate_hz))
    #     plot_interval = 1

    #     cap = cv2.VideoCapture(str(video_file))
    #     #ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    #     #ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frame_display = _resize(frame, width, height, 400)
    #         cv2.imshow("Video", frame_display)
    #         if cv2.waitKey(plot_interval) & 0xFF == ord('q'):
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
        # # Play video faster than plot updates (e.g., 60 FPS video, 30Hz plot)
        # print(f"\n--- Video @ 60 FPS / Plot @ {plot_update_rate_hz} Hz ---")
        # play_video_in_plot_controlled_speed(
        #     video_file,
        #     plot_interval_ms=plot_interval,
        #     target_video_fps=60.0,
        #     target_width=400
        # )

        # # Play video slower than plot updates (e.g., 15 FPS video, 30Hz plot)
        # print(f"\n--- Video @ 15 FPS / Plot @ {plot_update_rate_hz} Hz ---")
        # play_video_in_plot_controlled_speed(
        #     video_file,
        #     plot_interval_ms=plot_interval,
        #     target_video_fps=15.0,
        #     target_width=400
        # )

        # # Play video at original speed (or close to it)
        # cap_check = cv2.VideoCapture(str(video_file))
        # fps_check = cap_check.get(cv2.CAP_PROP_FPS) if cap_check.isOpened() else 30.0
        # cap_check.release()
        # print(f"\n--- Video @ Original FPS ({fps_check:.1f}) / Plot @ {plot_update_rate_hz} Hz ---")
        # play_video_in_plot_controlled_speed(
        #     video_file,
        #     plot_interval_ms=plot_interval,
        #     target_video_fps=plot_interval , # This is the FPS of the plot update rate
        #     target_width=400
        # )