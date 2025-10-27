#google collab
!pip install -q moviepy ultralytics gtts pydub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from ultralytics import YOLO
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from google.colab import files
import os
from IPython.display import Audio, display

# Fix deprecated aliases
np.int = int
np.bool = bool
np.float = float

# Configuration
LANE_SETTINGS = (80, 180, 5)
THRESHOLDS = (0.7, 0.4)
VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}
PEDESTRIANS = {'person'}
PRIORITY = {'person': 1, 'motorcycle': 2, 'car': 3, 'truck': 4, 'bus': 5}
COLORS = {'person': (255, 0, 0), 'vehicle': (0, 255, 255)}

# Global storage
confidence_list = []
detection_list = []
decision_list = []

# VOICE ALERT SYSTEM


class VoiceAlert:
    def __init__(self, fps=30):
        self.last_alert = ""
        self.cooldown = 0
        self.alerts_timeline = []
        self.fps = fps
        self.alert_counter = 0

    def speak(self, command, frame_num):
        """Generate voice alerts and store timeline"""
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        alerts = {
            'EMERGENCY STOP': 'Emergency! Stop immediately!',
            'STOP': 'Stop now! Pedestrian ahead!',
            'BRAKE': 'Brake! Vehicle too close!',
            'SLOW DOWN': 'Slow down. Poor visibility.'
        }

        if command in alerts and command != self.last_alert:
            try:
                self.alert_counter += 1
                audio_file = f'alert_{self.alert_counter}.mp3'

                tts = gTTS(text=alerts[command], lang='en', slow=False)
                tts.save(audio_file)

                timestamp = frame_num / self.fps
                self.alerts_timeline.append({
                    'time': timestamp,
                    'file': audio_file,
                    'command': command
                })

                self.last_alert = command
                self.cooldown = 60

                print(f"  {timestamp:.1f}s: {alerts[command]}")
                return audio_file

            except Exception as e:
                print(f"Voice error: {e}")
                return None

        return None

    def merge_audio_with_video(self, video_path, output_path):
        """Merge voice alerts into final video"""
        if not self.alerts_timeline:
            print("\n No voice alerts generated")
            os.rename(video_path, output_path)
            return

        try:
            print(f"\n Adding {len(self.alerts_timeline)} voice alerts to video...")

            video = VideoFileClip(video_path)
            audio_clips = []

            for alert in self.alerts_timeline:
                try:
                    audio = AudioFileClip(alert['file'])
                    audio = audio.set_start(alert['time'])
                    audio_clips.append(audio)
                except Exception as e:
                    print(f"  Skip alert at {alert['time']:.1f}s")

            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                final_video = video.set_audio(final_audio)

                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )

                print(f" Voice alerts embedded in video!")

                video.close()
                final_video.close()
                for clip in audio_clips:
                    clip.close()
            else:
                print(" No valid audio, saving without voice")
                os.rename(video_path, output_path)

        except Exception as e:
            print(f"   Audio merge failed: {e}")
            print("   Saving video without audio")
            try:
                os.rename(video_path, output_path)
            except:
                pass

# SPEED ESTIMATION


def estimate_speed(prev_detections, curr_detections, fps=30):
    """Calculate speed of approaching objects"""
    speed_warnings = []

    for curr in curr_detections:
        if curr['class'] not in VEHICLES:
            continue

        curr_center = curr['center']

        for prev in prev_detections:
            if prev['class'] == curr['class']:
                prev_center = prev['center']

                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                distance = np.sqrt(dx**2 + dy**2)

                speed = distance * fps
                approaching = dy > 3

                if approaching and speed > 80:
                    speed_warnings.append({
                        'class': curr['class'],
                        'speed': int(speed)
                    })
                break

    return speed_warnings

# LANE DETECTION USING HOUGH TRANSFORM,CANNY ENDGE FILTER

def detect_lanes(image):
    canny_low, canny_high, blur = LANE_SETTINGS

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(blur_img, canny_low, canny_high)

    height, width = image.shape[:2]
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 40,
                            minLineLength=50, maxLineGap=120)

    result = image.copy()
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)

            if 0.3 < abs(slope) < 2:
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])

        for lines_group in [left_lines, right_lines]:
            if lines_group:
                avg = np.mean(lines_group, axis=0).astype(int)
                cv2.line(result, (avg[0], avg[1]), (avg[2], avg[3]),
                        (0, 255, 0), 6)

    total = len(left_lines) + len(right_lines)
    has_both = len(left_lines) > 0 and len(right_lines) > 0
    confidence = min(np.clip(total / 8.0, 0, 1) * (1.2 if has_both else 0.7), 1.0)

    return (result, confidence)

# object detection
def detect_objects(image, model):
    """Object detection with threat assessment"""
    results = model(image, conf=0.45, verbose=False)

    detections = []
    result_img = image.copy()
    height = image.shape[0]

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = model.names[int(box.cls[0])]

            if cls in VEHICLES or cls in PEDESTRIANS:
                threat = 'HIGH' if y2 > height * 0.7 else \
                        'MED' if y2 > height * 0.5 else 'LOW'

                det_dict = {
                    'class': cls,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'priority': PRIORITY.get(cls, 5),
                    'threat': threat
                }
                detections.append(det_dict)

                color = (255, 0, 0) if threat == 'HIGH' else \
                       (255, 165, 0) if threat == 'MED' else (0, 255, 0)

                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result_img, f"{cls} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return (result_img, detections)

#decision engine
def make_decision(detections, confidence, speed_warnings):
    """Decision engine with speed awareness"""
    high_conf, low_conf = THRESHOLDS

    if confidence < 0.3:
        return ("EMERGENCY SLOW", "Critical: Poor visibility")

    if confidence < low_conf:
        return ("SLOW DOWN", f"Low confidence ({confidence:.2f})")

    if speed_warnings:
        fastest = max(speed_warnings, key=lambda x: x['speed'])
        if fastest['speed'] > 150:
            return ("BRAKE", f"Fast {fastest['class']} approaching!")

    high_peds = [d for d in detections
                 if d['class'] in PEDESTRIANS and d['threat'] == 'HIGH']
    if high_peds:
        return ("EMERGENCY STOP", "Pedestrian in path!")

    pedestrians = [d for d in detections if d['class'] in PEDESTRIANS]
    if pedestrians:
        return ("STOP", "Pedestrian detected")

    high_vehicles = [d for d in detections
                     if d['class'] in VEHICLES and d['threat'] == 'HIGH']
    if high_vehicles:
        return ("BRAKE", "Vehicle too close")

    med_vehicles = [d for d in detections
                    if d['class'] in VEHICLES and d['threat'] == 'MED']
    if med_vehicles:
        return ("CAUTION", f"{len(med_vehicles)} vehicle(s) ahead")

    return ("CONTINUE", "Path clear")

# LOGGING & VISUALIZATION

def create_log_dataframe(detection_list):
    """Create detection statistics"""
    df = pd.DataFrame(detection_list)

    if not df.empty:
        print("\n[PANDAS] Detection Statistics:")
        print(f"  Total: {len(df)}")
        print("  By class:")
        for cls, count in df['class'].value_counts().items():
            print(f"    {cls}: {count}")
        print(f"  Avg confidence: {df['confidence'].mean():.3f}")

    return df


def smooth_confidence(confidence_list):
    """Smooth confidence using Savitzky-Golay filter"""
    if len(confidence_list) < 5:
        return confidence_list

    conf_array = np.array(confidence_list)
    window = min(9, len(conf_array))
    if window % 2 == 0:
        window -= 1

    return signal.savgol_filter(conf_array, window, 2)

def create_simple_dashboard(confidence_list, detection_list, decision_list):
    """Generate analysis dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ADAS Analysis Dashboard', fontsize=14, fontweight='bold')

    # Confidence trend
    if confidence_list:
        conf_array = np.array(confidence_list)
        frames = np.arange(len(conf_array))

        axes[0, 0].plot(frames, conf_array, 'b-', alpha=0.4, label='Raw')
        if len(conf_array) >= 5:
            smoothed = smooth_confidence(confidence_list)
            axes[0, 0].plot(frames, smoothed, 'r-', linewidth=2, label='Smoothed')

        axes[0, 0].set_title('Lane Confidence Trend')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Decision distribution
    if decision_list:
        decision_counts = {}
        for cmd, _ in decision_list:
            decision_counts[cmd] = decision_counts.get(cmd, 0) + 1

        axes[0, 1].bar(decision_counts.keys(), decision_counts.values(),
                       color='steelblue')
        axes[0, 1].set_title('Decision Distribution')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # Confidence histogram
    if confidence_list:
        conf_array = np.array(confidence_list)
        axes[1, 0].hist(conf_array, bins=15, color='green',
                        alpha=0.7, edgecolor='black')
        mean = np.mean(conf_array)
        axes[1, 0].axvline(mean, color='red', linestyle='--',
                          label=f'Mean: {mean:.2f}')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

    # Detection by class
    if detection_list:
        df = pd.DataFrame(detection_list)
        df['class'].value_counts().plot(kind='barh', ax=axes[1, 1],
                                        color='coral')
        axes[1, 1].set_title('Detections by Class')
        axes[1, 1].set_xlabel('Count')

    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    print("\n✓ Dashboard saved: dashboard.png")
    plt.close()

# LEVEL FUNCTIONS


def level_1_lane_only(image_path):
    """Level 1: Lane detection"""
    print("\n")
    print("LEVEL 1: LANE DETECTION")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, confidence = detect_lanes(image_rgb)

    cv2.putText(result, f"Confidence: {confidence:.2f}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(result)
    axes[1].set_title(f"Lane Detection (Conf: {confidence:.2f})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('level1_output.png', dpi=150)
    print("✓ Saved: level1_output.png")

def level_2_objects_only(image_path):
    """Level 2: Object detection"""
    print("\n" )
    print("LEVEL 2: OBJECT DETECTION")


    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = YOLO('yolov8n.pt')
    result, detections = detect_objects(image_rgb, model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(result)
    axes[1].set_title(f"Objects: {len(detections)}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('level2_output.png', dpi=150)
    print(f"✓ Detected {len(detections)} objects")

def level_3_complete_image(image_path):
    """Level 3: Complete system"""
    print("\n")
    print("LEVEL 3: COMPLETE SYSTEM")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = YOLO('yolov8n.pt')

    lane_img, confidence = detect_lanes(image_rgb)
    detected_img, detections = detect_objects(lane_img, model)
    command, reason = make_decision(detections, confidence, [])

    final = detected_img.copy()
    overlay = final.copy()
    cv2.rectangle(overlay, (0, 0), (final.shape[1], 85), (0, 0, 0), -1)
    final = cv2.addWeighted(final, 0.7, overlay, 0.3, 0)

    color = (255, 0, 0) if "STOP" in command else \
            (255, 165, 0) if "SLOW" in command or "CAUTION" in command else (0, 255, 0)

    cv2.putText(final, command, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(final, reason, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(final)
    axes[1].set_title("Complete ADAS")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('level3_output.png', dpi=150)
    print(f"✓ Command: {command} | Reason: {reason}")

# LEVEL 4: VIDEO WITH VOICE & SPEED

def level_4_video(video_path, output_path='adas_output.mp4'):
    """Enhanced video processing with voice and speed detection"""
    print("\n")
    print("LEVEL 4: ENHANCED VIDEO PROCESSING")


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Error: Cannot open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f" Video: {width}x{height}, {fps} FPS, {total} frames")
    print(f" Voice alerts: ENABLED")
    print(f" Speed detection: ENABLED\n")

    # Temporary video without audio
    temp_video = 'temp_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    model = YOLO('yolov8n.pt')
    voice = VoiceAlert(fps)

    conf_buffer = []
    prev_detections = []
    buffer_size = 5
    frame_num = 0
    speed_event_count = 0

    print("Processing frames...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            lane_img, confidence = detect_lanes(frame_rgb)
            detected_img, detections = detect_objects(lane_img, model)

            # Speed estimation
            speed_warnings = estimate_speed(prev_detections, detections, fps)
            prev_detections = detections.copy()

            # Smooth confidence
            conf_buffer.append(confidence)
            if len(conf_buffer) > buffer_size:
                conf_buffer.pop(0)
            smooth_conf = np.mean(conf_buffer)

            # Decision
            command, reason = make_decision(detections, smooth_conf, speed_warnings)

            # Voice alerts (check every 10 frames)
            if frame_num % 10 == 0:
                voice.speak(command, frame_num)

            # Store data
            confidence_list.append(smooth_conf)
            detection_list.extend(detections)
            decision_list.append((command, reason))

            # Create overlay
            final = detected_img.copy()
            overlay = final.copy()
            cv2.rectangle(overlay, (0, 0), (width, 125), (0, 0, 0), -1)
            final = cv2.addWeighted(final, 0.65, overlay, 0.35, 0)

            # Color coding
            if "STOP" in command or "EMERGENCY" in command:
                color = (255, 0, 0)
            elif "SLOW" in command or "CAUTION" in command or "BRAKE" in command:
                color = (255, 165, 0)
            else:
                color = (0, 255, 0)

            # Command display
            cv2.putText(final, command, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            cv2.putText(final, reason, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Confidence indicator
            conf_color = (0, 255, 0) if smooth_conf > 0.7 else \
                        (255, 165, 0) if smooth_conf > 0.4 else (255, 0, 0)
            cv2.putText(final, f"Conf: {smooth_conf:.2f}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

            # Speed warnings
            if speed_warnings:
                speed_event_count += 1
                warning = speed_warnings[0]
                cv2.putText(final, f"⚡ Fast {warning['class']}: {warning['speed']}px/s",
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Frame info
            cv2.putText(final, f"{frame_num}/{total}", (width-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(final, f"Objects: {len(detections)}", (width-150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Write frame
            final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            out.write(final_bgr)

            if frame_num % 30 == 0:
                pct = (frame_num / total) * 100
                print(f"  {frame_num}/{total} ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n Processing interrupted")
    finally:
        cap.release()
        out.release()

    print(f"\n Video processing complete")
    print(f" Processed {frame_num} frames")
    print(f" Speed events: {speed_event_count}")

    # Merge voice alerts into video
    voice.merge_audio_with_video(temp_video, output_path)

    # Cleanup temp file
    try:
        if os.path.exists(temp_video) and temp_video != output_path:
            os.remove(temp_video)
    except:
        pass

    # Generate logs
    if detection_list:
        df = create_log_dataframe(detection_list)
        if not df.empty:
            df.to_csv('detection_log.csv', index=False)
            print(" CSV saved: detection_log.csv")

    create_simple_dashboard(confidence_list, detection_list, decision_list)

    # Summary statistics
    print("\n")
    print("SUMMARY STATISTICS")

    if confidence_list:
        conf_array = np.array(confidence_list)
        print(f"\n[NumPy] Confidence:")
        print(f"  Mean: {np.mean(conf_array):.3f}")
        print(f"  Std: {np.std(conf_array):.3f}")
        print(f"  Range: {np.min(conf_array):.3f} - {np.max(conf_array):.3f}")

    print(f"\n[Lists] Data Collected:")
    print(f"  Confidence points: {len(confidence_list)}")
    print(f"  Total detections: {len(detection_list)}")
    print(f"  Decisions made: {len(decision_list)}")

    decision_counts = {}
    for cmd, _ in decision_list:
        decision_counts[cmd] = decision_counts.get(cmd, 0) + 1

    print(f"\n[Dict] Decision Distribution:")
    for cmd, count in sorted(decision_counts.items(),
                             key=lambda x: x[1], reverse=True):
        pct = (count / len(decision_list)) * 100
        print(f"  {cmd}: {count} ({pct:.1f}%)")

    print(f"\n✓ Final video with voice: {output_path}")

# MAIN

if __name__ == "__main__":
    print("Enter your choice:")
    print("""
    1 - Lane detection (image)
    2 - Object detection (image)
    3 - Complete system (image)
    4 - Video with voice & speed
    """)

    try:
        choice = int(input("Choice (1-4): "))
    except ValueError:
        print(" Invalid input")
        exit()

    if choice in [1, 2, 3]:
        print("\nUpload IMAGE:")
        uploaded = files.upload()
        path = list(uploaded.keys())[0]

        if choice == 1:
            level_1_lane_only(path)
        elif choice == 2:
            level_2_objects_only(path)
        else:
            level_3_complete_image(path)

    elif choice == 4:
        print("\nUpload VIDEO:")
        uploaded = files.upload()
        path = list(uploaded.keys())[0]
        level_4_video(path, "adas_enhanced_output.mp4")
        print("\n DONE! Play the video to hear voice alerts!")

    else:
        print(" Invalid choice")
