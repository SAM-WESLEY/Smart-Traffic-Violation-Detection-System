import cv2
import argparse
import threading
import time
from flask import Flask, jsonify, Response
from modules.red_light_detector import RedLightDetector
from modules.helmet_detector     import HelmetDetector
from modules.plate_reader        import PlateReader
from modules.violation_logger    import ViolationLogger

# ── Config ─────────────────────────────────────────────────────────────────────
FLASK_PORT   = 5000
CROWD_LIMIT  = 0   # unused here but kept for consistency

# ── Shared State ───────────────────────────────────────────────────────────────
state = {
    "signal":            "GREEN",
    "total_violations":  0,
    "red_light_count":   0,
    "helmet_count":      0,
    "last_plate":        "—",
    "fps":               0.0,
    "running":           False,
}

# ── Modules ────────────────────────────────────────────────────────────────────
red_light = RedLightDetector()
helmet    = HelmetDetector()
plate     = PlateReader()
logger    = ViolationLogger()

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

def generate_frames(source):
    cap = cv2.VideoCapture(source)
    t_prev = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ── Red Light Detection ────────────────────────────────────────────
        signal, crossed = red_light.detect(frame)
        state["signal"] = signal
        if crossed:
            plate_text = plate.read(frame)
            state["last_plate"] = plate_text or "Unknown"
            logger.log("Red Light", plate_text, frame)
            state["red_light_count"] += 1
            state["total_violations"] += 1

        # ── Helmet Detection ───────────────────────────────────────────────
        frame, violations = helmet.detect(frame)
        for v in violations:
            plate_text = plate.read(frame)
            state["last_plate"] = plate_text or "Unknown"
            logger.log("No Helmet", plate_text, frame)
            state["helmet_count"] += 1
            state["total_violations"] += 1

        # ── FPS ────────────────────────────────────────────────────────────
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        state["fps"] = round(fps, 1)

        # ── HUD ────────────────────────────────────────────────────────────
        sig_color = (0, 0, 255) if signal == "RED" else (0, 200, 0)
        cv2.putText(frame, f"Signal : {signal}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, sig_color, 2)
        cv2.putText(frame, f"Violations: {state['total_violations']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(frame, f"Last Plate: {state['last_plate']}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "S: toggle signal  R: reset  ESC: quit",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Stream frame ───────────────────────────────────────────────────
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            state["running"] = False
        elif key in (ord('s'), ord('S')):
            state["signal"] = "RED" if state["signal"] == "GREEN" else "GREEN"
            red_light.manual_signal = state["signal"]
        elif key in (ord('r'), ord('R')):
            state["total_violations"] = 0
            state["red_light_count"]  = 0
            state["helmet_count"]     = 0

    cap.release()


@app.route('/')
def dashboard():
    return open('dashboard/index.html').read()


@app.route('/video_feed')
def video_feed():
    source = app.config.get('SOURCE', 0)
    return Response(generate_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(state)


@app.route('/violations')
def violations():
    return jsonify(logger.get_recent(20))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0',
                        help='Camera index (0) or video file path')
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    state["running"] = True
    app.config['SOURCE'] = source
    app.run(host='0.0.0.0', port=FLASK_PORT)
