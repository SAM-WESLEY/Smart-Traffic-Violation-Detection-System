import cv2
import numpy as np

# Stop line position (y-coordinate as fraction of frame height)
STOP_LINE_Y = 0.6


class RedLightDetector:
    def __init__(self):
        self.manual_signal = None
        self.signal_state  = "GREEN"
        self.frame_count   = 0

    def detect_signal_color(self, frame):
        """Detect traffic signal color using HSV segmentation."""
        h, w = frame.shape[:2]
        # Check top-right region for signal
        roi = frame[0:h//3, w//2:w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red color mask (two ranges for red in HSV)
        red1 = cv2.inRange(hsv, np.array([0,  120, 70]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170,120, 70]), np.array([180,255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        # Green color mask
        green_mask = cv2.inRange(hsv,
                                  np.array([40, 50, 50]),
                                  np.array([90, 255, 255]))

        red_pixels   = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        if red_pixels > 500 and red_pixels > green_pixels:
            return "RED"
        return "GREEN"

    def check_stop_line_crossed(self, frame, boxes):
        """Check if any vehicle bounding box crosses the stop line."""
        h = frame.shape[0]
        stop_y = int(h * STOP_LINE_Y)

        # Draw stop line
        cv2.line(frame, (0, stop_y), (frame.shape[1], stop_y),
                 (0, 0, 255), 2)
        cv2.putText(frame, "STOP LINE", (10, stop_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for (x1, y1, x2, y2) in boxes:
            # Vehicle bottom edge crossing stop line
            if y2 > stop_y:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                return True
        return False

    def detect(self, frame):
        """Main detection — returns (signal_state, violation_occurred)."""
        self.frame_count += 1

        # Use manual override if set
        if self.manual_signal:
            self.signal_state = self.manual_signal
        elif self.frame_count % 10 == 0:
            self.signal_state = self.detect_signal_color(frame)

        violation = False
        if self.signal_state == "RED":
            # Simple motion-based crossing detection (demo)
            # In production: use YOLO vehicle detections
            h, w = frame.shape[:2]
            stop_y = int(h * STOP_LINE_Y)
            cv2.line(frame, (0, stop_y), (w, stop_y), (0, 0, 255), 2)
            # Demo: simulate occasional violation
            if self.frame_count % 150 == 0:
                violation = True

        return self.signal_state, violation
