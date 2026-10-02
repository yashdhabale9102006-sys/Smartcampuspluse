import cv2
import time
import requests
import os
from plyer import notification
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
# Timer duration before alert triggers (Set to 10s for quick testing/demo; change to 300 for 5 mins)
EMPTY_THRESHOLD_SECONDS = 5

# Optional: Add your Telegram credentials to get mobile alerts
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

CONFIDENCE_THRESHOLD = 0.5  # Detection sensitivity (50%)
# =======================================================

def send_os_notification(elapsed_seconds):
    """Triggers a native Windows/macOS pop-up notification on the laptop."""
    try:
        notification.notify(
            title="🚨 CLASSROOM ENERGY ALERT",
            message=f"Classroom empty for {int(elapsed_seconds)}s! Please switch off power.",
            app_name="Classroom Monitor",
            timeout=8
        )
        print("[SUCCESS] Local OS notification banner displayed!")
    except Exception as e:
        print(f"[ERROR] Could not display OS notification: {e}")

def send_telegram_alert(image_path, elapsed_seconds):
    """Sends a push notification with snapshot to Telegram (if configured)."""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        return  # Skip if credentials are not filled in

    caption = (
        f"🚨 *ALERT: Classroom Energy Warning*\n\n"
        f"Classroom empty for *{int(elapsed_seconds)} seconds*.\n"
        f"Please request staff to switch off lights and appliances."
    )
    
    msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": caption, "parse_mode": "Markdown"}
    
    try:
        requests.post(msg_url, data=payload, timeout=5)
        if os.path.exists(image_path):
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo}, timeout=5)
        print("[SUCCESS] Telegram push notification sent!")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram notification: {e}")

def main():
    print("Loading YOLOv8 Human Detection Model...")
    # Downloads 'yolov8n.pt' automatically on first run (~6MB)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Laptop webcam could not be accessed.")
        return

    empty_start_time = None
    alert_triggered = False

    print("\n--- SYSTEM ACTIVE ---")
    print(f"Alert will trigger after {EMPTY_THRESHOLD_SECONDS} seconds of zero occupancy.")
    print("Press 'q' in the camera window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab webcam frame.")
            break

        # Run detection specifically filtering for humans (class 0)
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
        person_count = 0
        annotated_frame = frame.copy()

        for result in results:
            person_count = len(result.boxes)
            annotated_frame = result.plot()  # Draws bounding boxes automatically

        # Core Occupancy & Timer Logic
        if person_count > 0:
            # Humans present -> reset timer and alert lock
            empty_start_time = None
            alert_triggered = False
            status_text = f"Status: OCCUPIED ({person_count} person)"
            timer_text = "Timer: Standby"
            status_color = (0, 255, 0)  # Green
        else:
            # Room empty -> start/update timer
            if empty_start_time is None:
                empty_start_time = time.time()

            elapsed_seconds = time.time() - empty_start_time
            remaining = max(0, EMPTY_THRESHOLD_SECONDS - int(elapsed_seconds))

            status_text = "Status: EMPTY CLASSROOM"
            timer_text = f"Countdown to Alert: {remaining}s"
            status_color = (0, 0, 255)  # Red

            # Trigger condition reached
            if elapsed_seconds >= EMPTY_THRESHOLD_SECONDS and not alert_triggered:
                snapshot_file = "empty_room_snapshot.jpg"
                cv2.imwrite(snapshot_file, annotated_frame)
                
                print("\n[TRIGGER] Threshold reached! Sending alerts...")
                send_os_notification(elapsed_seconds)
                send_telegram_alert(snapshot_file, elapsed_seconds)
                alert_triggered = True

        # Overlay visual status interface
        cv2.putText(annotated_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated_frame, timer_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw alert banner on webcam view if triggered
        if alert_triggered:
            cv2.rectangle(annotated_frame, (0, 420), (annotated_frame.shape[1], 480), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "ALERT SENT! SWITCH OFF POWER", (20, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display camera feed window
        cv2.imshow("Classroom Power Automation System", annotated_frame)

        # Press 'q' to stop execution
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()