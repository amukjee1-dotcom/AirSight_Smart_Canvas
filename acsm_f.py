import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import time
import math
import threading
import queue

# Smart handwritten mathematics is kept in sm_f.py so the OCR,
# preprocessing, calculus/equation solving, and graphing logic has one implementation.
from sm_f import (
    solve_handwriting,
    graph_handwriting,
    prepare_graph_data,
    build_graph_payload_from_latex,
    show_graph_window,
)

math_panel_lock = threading.Lock()
math_panel_text = []
math_panel_until = 0.0
solving_in_progress = False
should_quit = False
pending_graph_queue = queue.Queue()

TOP_BAR_HEIGHT = 116
MAX_HISTORY = 25


def show_math_message(lines, seconds=50):
    global math_panel_text, math_panel_until
    wrapped_lines = []
    for line in lines:
        text = str(line).replace("\r", " ").replace("\n", " ").strip()
        safe = text.encode("ascii", "replace").decode("ascii")
        while len(safe) > 84:
            wrapped_lines.append(safe[:84])
            safe = "  " + safe[84:]
        wrapped_lines.append(safe)
    with math_panel_lock:
        math_panel_text = wrapped_lines[:14]
        math_panel_until = time.monotonic() + seconds


def draw_math_panel(frame):
    now = time.monotonic()
    with math_panel_lock:
        if now > math_panel_until or not math_panel_text:
            return
        lines_to_draw = list(math_panel_text)

    line_step = 26
    box_x1, box_y1 = 25, TOP_BAR_HEIGHT + 12
    box_x2 = min(frame.shape[1] - 25, 920)
    box_y2 = min(frame.shape[0] - 55, box_y1 + 44 + len(lines_to_draw) * line_step)

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

    cv2.putText(
        frame,
        "SMART MATH SOLVER (STEPS + ANSWER)",
        (box_x1 + 18, box_y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (0, 255, 255),
        2,
    )

    y = box_y1 + 54
    for idx, line in enumerate(lines_to_draw):
        if line.startswith("Answer:"):
            color = (0, 255, 120)
            thickness = 2
        elif line.startswith("Step ") or line.startswith("  "):
            color = (220, 245, 255)
            thickness = 1
        elif line.startswith("Recognized:"):
            color = (0, 220, 255)
            thickness = 1
        else:
            color = (255, 255, 255)
            thickness = 1

        cv2.putText(
            frame,
            line,
            (box_x1 + 18, y + idx * line_step),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            color,
            thickness,
        )


# ============================================================
# AIR CANVAS ADVANCED - TASKS API (PAINT STUDIO + SMART MATH)
# ============================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("MediaPipe Hand Landmarker model not found. Downloading...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        print("Could not download model automatically:", e)
        raise SystemExit(1)


ensure_model()

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

hand_detector = vision.HandLandmarker.create_from_options(options)

math_canvas = None
art_canvas = None
undo_stack = []
redo_stack = []

active_layer = "MATH"
active_tool = "BRUSH"
shape_filled = False
draw_color = (0, 0, 255)
active_brush_name = "MEDIUM"
brush_thickness = 7
eraser_thickness = 45

colors = [
    [10, 8, 60, 52, (0, 0, 255), "RED"],
    [68, 8, 118, 52, (0, 255, 0), "GREEN"],
    [126, 8, 176, 52, (255, 0, 0), "BLUE"],
    [184, 8, 234, 52, (0, 255, 255), "YELLOW"],
    [242, 8, 292, 52, (255, 255, 0), "CYAN"],
    [300, 8, 350, 52, (255, 0, 255), "MAGENTA"],
    [358, 8, 408, 52, (0, 165, 255), "ORANGE"],
    [416, 8, 466, 52, (255, 255, 255), "WHITE"],
]

brush_sizes = [
    [474, 8, 542, 52, 2, "PENCIL"],
    [548, 8, 622, 52, 7, "MEDIUM"],
    [628, 8, 696, 52, 18, "THICK"],
]

fill_mode_button = [704, 8, 840, 52, "FILL_MODE"]
layer_mode_button = [848, 8, 995, 52, "LAYER_MODE"]

tool_buttons = [
    [10, 62, 74, 106, "BRUSH"],
    [80, 62, 128, 106, "FILL"],
    [134, 62, 186, 106, "LINE"],
    [192, 62, 244, 106, "RECT"],
    [250, 62, 314, 106, "CIRCLE"],
    [320, 62, 394, 106, "ELLIPSE"],
    [400, 62, 502, 106, "EYEDROPPER"],
    [508, 62, 584, 106, "ERASER"],
    [590, 62, 650, 106, "UNDO"],
    [656, 62, 716, 106, "REDO"],
    [722, 62, 814, 106, "CLEAR"],
    [820, 62, 954, 106, "SOLVE"],
    [960, 62, 1036, 106, "GRAPH"],
    [1042, 62, 1112, 106, "QUIT"],
]

xp, yp = 0, 0
smooth_x, smooth_y = 0.0, 0.0
ALPHA = 0.40

click_cooldown = 0
pinch_was_closed = False
suppress_drawing_until_release = False
shape_start_pt = None
stroke_snapshot_saved = False

DRAW_PINCH_CLOSE = 0.30
DRAW_PINCH_RELEASE = 0.40


def get_target_canvas():
    return math_canvas if active_layer == "MATH" else art_canvas


def get_combined_canvas():
    return cv2.bitwise_or(art_canvas, math_canvas)


def get_math_ocr_canvas():
    if math_canvas is not None and np.any(math_canvas > 0):
        return math_canvas.copy()
    return get_combined_canvas()


def save_history_snapshot():
    global undo_stack, redo_stack
    if math_canvas is None or art_canvas is None:
        return
    undo_stack.append((math_canvas.copy(), art_canvas.copy()))
    if len(undo_stack) > MAX_HISTORY:
        undo_stack.pop(0)
    redo_stack.clear()


def perform_undo():
    global math_canvas, art_canvas, shape_start_pt
    if not undo_stack:
        print("[History] Nothing to undo.")
        return
    redo_stack.append((math_canvas.copy(), art_canvas.copy()))
    prev_math, prev_art = undo_stack.pop()
    math_canvas[:] = prev_math
    art_canvas[:] = prev_art
    shape_start_pt = None
    print(f"[History] Undo applied ({len(undo_stack)} steps left).")


def perform_redo():
    global math_canvas, art_canvas, shape_start_pt
    if not redo_stack:
        print("[History] Nothing to redo.")
        return
    undo_stack.append((math_canvas.copy(), art_canvas.copy()))
    next_math, next_art = redo_stack.pop()
    math_canvas[:] = next_math
    art_canvas[:] = next_art
    shape_start_pt = None
    print(f"[History] Redo applied ({len(redo_stack)} redo steps left).")


def render_shape(target_img, tool_name, p1, p2, color, thickness, filled):
    x1, y1 = p1
    x2, y2 = p2
    draw_thick = -1 if filled else max(1, thickness)

    if tool_name == "LINE":
        cv2.line(target_img, (x1, y1), (x2, y2), color, max(1, thickness), cv2.LINE_AA)
    elif tool_name == "RECT":
        cv2.rectangle(target_img, (x1, y1), (x2, y2), color, draw_thick, cv2.LINE_AA)
    elif tool_name == "CIRCLE":
        radius = int(round(math.hypot(x2 - x1, y2 - y1)))
        if radius > 1:
            cv2.circle(target_img, (x1, y1), radius, color, draw_thick, cv2.LINE_AA)
    elif tool_name == "ELLIPSE":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = abs(x2 - x1) // 2
        ry = abs(y2 - y1) // 2
        if rx > 1 and ry > 1:
            cv2.ellipse(target_img, (cx, cy), (rx, ry), 0, 0, 360, color, draw_thick, cv2.LINE_AA)


def flood_fill_canvas(target_canvas, seed_pt, new_color):
    h, w = target_canvas.shape[:2]
    sx = max(0, min(w - 1, int(seed_pt[0])))
    sy = max(TOP_BAR_HEIGHT, min(h - 1, int(seed_pt[1])))

    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[: TOP_BAR_HEIGHT + 1, :] = 1

    current_bgr = tuple(int(c) for c in target_canvas[sy, sx])
    fill_bgr = tuple(int(c) for c in new_color)
    if fill_bgr == (0, 0, 0):
        fill_bgr = (10, 10, 10)
    if current_bgr == fill_bgr:
        return False

    save_history_snapshot()
    cv2.floodFill(
        target_canvas,
        mask,
        (sx, sy),
        fill_bgr,
        loDiff=(28, 28, 28),
        upDiff=(28, 28, 28),
        flags=4 | cv2.FLOODFILL_FIXED_RANGE,
    )
    return True


def draw_ui(frame):
    cv2.rectangle(frame, (0, 0), (1280, TOP_BAR_HEIGHT), (28, 28, 28), -1)
    cv2.line(frame, (0, TOP_BAR_HEIGHT), (1280, TOP_BAR_HEIGHT), (90, 90, 90), 1)

    for col in colors:
        x1, y1, x2, y2, color, _ = col
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if active_tool not in ("ERASER", "EYEDROPPER") and draw_color == color:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for bs in brush_sizes:
        x1, y1, x2, y2, _, name = bs
        is_sel = active_brush_name == name and active_tool != "ERASER"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 55, 55), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if is_sel else (140, 140, 140), 2 if is_sel else 1)
        cv2.putText(
            frame,
            name,
            (x1 + 5, y1 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0) if is_sel else (225, 225, 225),
            1,
        )

    fx1, fy1, fx2, fy2, _ = fill_mode_button
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 130, 180) if shape_filled else (65, 65, 65), -1)
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255) if shape_filled else (160, 160, 160), 2 if shape_filled else 1)
    mode_label = "SHAPE: SOLID" if shape_filled else "SHAPE: OUTLINE"
    cv2.putText(frame, mode_label, (fx1 + 6, fy1 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    lx1, ly1, lx2, ly2, _ = layer_mode_button
    is_math_layer = active_layer == "MATH"
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (140, 70, 180) if is_math_layer else (40, 120, 140), -1)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
    layer_label = "LAYER: MATH OCR" if is_math_layer else "LAYER: ART ONLY"
    cv2.putText(frame, layer_label, (lx1 + 6, ly1 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    for btn in tool_buttons:
        x1, y1, x2, y2, name = btn
        is_active = active_tool == name
        if name == "CLEAR":
            bg_col = (45, 45, 175)
            border_col = (220, 220, 220)
            label = "CLEAR ALL"
            font_scale = 0.38
            thick = 1
        elif name == "SOLVE":
            bg_col = (140, 85, 25) if not solving_in_progress else (180, 120, 30)
            border_col = (0, 255, 255) if solving_in_progress else (220, 220, 220)
            label = "SOLVING..." if solving_in_progress else "SOLVE+GRAPH"
            font_scale = 0.37
            thick = 2 if solving_in_progress else 1
        elif name == "GRAPH":
            bg_col = (150, 50, 120)
            border_col = (220, 220, 220)
            label = "GRAPH"
            font_scale = 0.40
            thick = 1
        elif name == "QUIT":
            bg_col = (30, 30, 175)
            border_col = (0, 255, 255)
            label = "QUIT"
            font_scale = 0.42
            thick = 2
        elif name == "UNDO":
            bg_col = (85, 85, 110) if undo_stack else (45, 45, 45)
            border_col = (200, 200, 200) if undo_stack else (100, 100, 100)
            label = "UNDO"
            font_scale = 0.38
            thick = 1
        elif name == "REDO":
            bg_col = (85, 85, 110) if redo_stack else (45, 45, 45)
            border_col = (200, 200, 200) if redo_stack else (100, 100, 100)
            label = "REDO"
            font_scale = 0.38
            thick = 1
        elif name == "ERASER" and is_active:
            bg_col = (0, 0, 200)
            border_col = (0, 255, 255)
            label = "ERASING"
            font_scale = 0.38
            thick = 2
        elif is_active:
            bg_col = (0, 150, 90)
            border_col = (0, 255, 255)
            label = name
            font_scale = 0.38
            thick = 2
        else:
            bg_col = (75, 75, 75)
            border_col = (180, 180, 180)
            label = name
            font_scale = 0.38
            thick = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 2 if is_active else 1)
        cv2.putText(frame, label, (x1 + 5, y1 + 29), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)

    cv2.circle(frame, (1205, 56), 24, draw_color, -1)
    cv2.circle(frame, (1205, 56), 24, (255, 255, 255), 2)
    cv2.putText(frame, active_layer, (1183, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)


def clear_canvas():
    global math_canvas, art_canvas, shape_start_pt
    save_history_snapshot()
    shape_start_pt = None
    if math_canvas is not None:
        math_canvas[:] = 0
    if art_canvas is not None:
        art_canvas[:] = 0


def _solve_in_background(math_canvas_copy):
    global solving_in_progress
    try:
        latex_text, answer, steps = solve_handwriting(math_canvas_copy)
        print("\n========== SMART MATH SOLVER ==========")
        print("Recognized:", latex_text)
        for i, step in enumerate(steps, start=1):
            print(f"Step {i}: {step}")
        print("Answer:", answer)
        print("=======================================\n")

        panel_lines = [f"Recognized: {latex_text}"]
        for i, step in enumerate(steps, start=1):
            clean_step = step if step.lower().startswith("step ") else f"Step {i}: {step}"
            panel_lines.append(clean_step)
        panel_lines.append(f"Answer: {answer}")

        show_math_message(panel_lines, seconds=55)

        # Build graph payload from already-recognized LaTeX (no extra API request needed)
        try:
            graph_payload = build_graph_payload_from_latex(latex_text)
            pending_graph_queue.put(graph_payload)
        except Exception as graph_err:
            print("[Auto-Graph Note]", graph_err)
    except Exception as error:
        print("[Math Solver Error]", error)
        show_math_message(["Solver error:", str(error)], seconds=35)
    finally:
        solving_in_progress = False


def trigger_solve_steps_graph():
    global solving_in_progress
    if solving_in_progress:
        print("[Debug] Solve requested while already solving; ignoring duplicate request.")
        return
    print("[Debug] Starting solver + auto-graph on isolated MATH layer...")
    solving_in_progress = True
    show_math_message(["Solving algebra/calculus + steps with Gemini..."], seconds=60)
    threading.Thread(
        target=_solve_in_background,
        args=(get_math_ocr_canvas(),),
        daemon=True,
    ).start()


def trigger_graph():
    print("[Debug] Opening graph from isolated MATH layer...")
    try:
        latex_text = graph_handwriting(get_math_ocr_canvas())
        print("[Graph Plotter] Graphed:", latex_text)
        show_math_message([f"Graph opened for: {latex_text}"])
    except Exception as error:
        print("[Graph Plotter Error]", error)
        show_math_message(["Graph error:", str(error)])


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    hand_detector.close()
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_time = time.monotonic()
last_timestamp_ms = -1
print("[Air Canvas] S: Solve+Steps+Graph | G: Graph | L: Math/Art Layer | M: Solid/Outline | Z: Undo | Y: Redo | C: Clear | Q: Quit")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        camera_frame = frame.copy()

        if math_canvas is None or math_canvas.shape != frame.shape:
            math_canvas = np.zeros_like(frame)
            art_canvas = np.zeros_like(frame)

        h, w, _ = frame.shape
        preview_layer = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        results = hand_detector.detect_for_video(mp_image, timestamp_ms)

        draw_ui(frame)

        if click_cooldown > 0:
            click_cooldown -= 1

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            raw_x1, raw_y1 = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = float(raw_x1), float(raw_y1)
            else:
                smooth_x = ALPHA * raw_x1 + (1 - ALPHA) * smooth_x
                smooth_y = ALPHA * raw_y1 + (1 - ALPHA) * smooth_y

            x1, y1 = int(smooth_x), int(smooth_y)
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            tx, ty = max(0, min(w - 1, tx)), max(0, min(h - 1, ty))

            palm_width = math.hypot(
                (hand_landmarks[5].x - hand_landmarks[17].x) * w,
                (hand_landmarks[5].y - hand_landmarks[17].y) * h,
            )
            palm_width = max(palm_width, 1.0)
            pinch_dist = math.hypot(raw_x1 - tx, raw_y1 - ty)
            pinch_ratio = pinch_dist / palm_width

            pinch_now = pinch_ratio < DRAW_PINCH_RELEASE if pinch_was_closed else pinch_ratio < DRAW_PINCH_CLOSE
            pinch_just_started = pinch_now and not pinch_was_closed
            pinch_just_released = (not pinch_now) and pinch_was_closed

            if not pinch_now:
                suppress_drawing_until_release = False
                stroke_snapshot_saved = False

            if pinch_just_released and shape_start_pt is not None and active_tool in ("LINE", "RECT", "CIRCLE", "ELLIPSE"):
                end_pt = (x1, max(TOP_BAR_HEIGHT, y1))
                save_history_snapshot()
                render_shape(
                    get_target_canvas(),
                    active_tool,
                    shape_start_pt,
                    end_pt,
                    draw_color,
                    brush_thickness,
                    shape_filled,
                )
                shape_start_pt = None

            if y1 < TOP_BAR_HEIGHT and shape_start_pt is None:
                xp, yp = 0, 0
                cv2.circle(frame, (x1, y1), 8, (0, 255, 255), -1)

                if pinch_just_started and click_cooldown == 0:
                    suppress_drawing_until_release = True
                    click_cooldown = 12

                    for col in colors:
                        cx1, cy1, cx2, cy2, color, _ = col
                        if cx1 <= x1 <= cx2 and cy1 <= y1 <= cy2:
                            draw_color = color
                            xp, yp = 0, 0
                            if active_tool in ("ERASER", "EYEDROPPER"):
                                active_tool = "BRUSH"
                            break

                    for bs in brush_sizes:
                        bx1, by1, bx2, by2, thickness, name = bs
                        if bx1 <= x1 <= bx2 and by1 <= y1 <= by2:
                            brush_thickness = thickness
                            active_brush_name = name
                            if active_tool == "ERASER":
                                active_tool = "BRUSH"
                            break

                    fx1, fy1, fx2, fy2, _ = fill_mode_button
                    if fx1 <= x1 <= fx2 and fy1 <= y1 <= fy2:
                        shape_filled = not shape_filled

                    lx1, ly1, lx2, ly2, _ = layer_mode_button
                    if lx1 <= x1 <= lx2 and ly1 <= y1 <= ly2:
                        active_layer = "ART" if active_layer == "MATH" else "MATH"

                    for btn in tool_buttons:
                        ax1, ay1, ax2, ay2, name = btn
                        if ax1 <= x1 <= ax2 and ay1 <= y1 <= ay2:
                            if name == "CLEAR":
                                clear_canvas()
                            elif name == "SOLVE":
                                trigger_solve_steps_graph()
                            elif name == "GRAPH":
                                trigger_graph()
                            elif name == "QUIT":
                                print("[Air Canvas] QUIT button pinched; exiting...")
                                should_quit = True
                            elif name == "UNDO":
                                perform_undo()
                            elif name == "REDO":
                                perform_redo()
                            else:
                                active_tool = name
                                if name in ("FILL", "LINE", "RECT", "CIRCLE", "ELLIPSE"):
                                    active_layer = "ART"
                                elif name == "BRUSH":
                                    active_layer = "MATH"
                                shape_start_pt = None
                                xp, yp = 0, 0
                            break

            elif active_tool == "EYEDROPPER":
                xp, yp = 0, 0
                index_mcp = hand_landmarks[5]
                mcp_x, mcp_y = int(index_mcp.x * w), int(index_mcp.y * h)
                dx, dy = x1 - mcp_x, y1 - mcp_y
                length = max(1.0, math.hypot(dx, dy))
                offset = 48
                sample_x = min(w - 6, max(5, int(x1 + (dx / length) * offset)))
                sample_y = min(h - 6, max(TOP_BAR_HEIGHT + 5, int(y1 + (dy / length) * offset)))

                cv2.circle(frame, (sample_x, sample_y), 10, (0, 255, 255), 2)
                cv2.circle(frame, (sample_x, sample_y), 3, (255, 255, 255), -1)
                cv2.line(frame, (x1, y1), (sample_x, sample_y), (0, 255, 255), 1)

                if pinch_just_started and click_cooldown == 0:
                    suppress_drawing_until_release = True
                    click_cooldown = 15
                    y_start, y_end = max(0, sample_y - 3), min(h, sample_y + 4)
                    x_start, x_end = max(0, sample_x - 3), min(w, sample_x + 4)
                    patch = camera_frame[y_start:y_end, x_start:x_end]
                    if patch.size:
                        median_bgr = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
                        draw_color = tuple(int(v) for v in median_bgr)
                    active_tool = "BRUSH"

            elif active_tool == "FILL":
                xp, yp = 0, 0
                cv2.circle(frame, (x1, y1), 10, draw_color, -1)
                cv2.circle(frame, (x1, y1), 12, (255, 255, 255), 2)
                cv2.putText(frame, f"FILL ({active_layer})", (x1 + 15, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

                if pinch_just_started and not suppress_drawing_until_release and click_cooldown == 0:
                    suppress_drawing_until_release = True
                    click_cooldown = 15
                    flood_fill_canvas(get_target_canvas(), (x1, y1), draw_color)

            elif active_tool in ("LINE", "RECT", "CIRCLE", "ELLIPSE"):
                xp, yp = 0, 0
                clamped_y = max(TOP_BAR_HEIGHT, y1)
                if pinch_just_started and not suppress_drawing_until_release:
                    shape_start_pt = (x1, clamped_y)
                elif pinch_now and shape_start_pt is not None:
                    render_shape(
                        preview_layer,
                        active_tool,
                        shape_start_pt,
                        (x1, clamped_y),
                        draw_color,
                        brush_thickness,
                        shape_filled,
                    )
                    cv2.circle(frame, shape_start_pt, 5, (0, 255, 255), -1)
                    cv2.putText(
                        frame,
                        f"{active_tool} ({active_layer})",
                        (x1 + 15, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.circle(frame, (x1, y1), 8, (0, 255, 255), 2)

            elif pinch_now and not suppress_drawing_until_release:
                if not stroke_snapshot_saved:
                    save_history_snapshot()
                    stroke_snapshot_saved = True

                current_color = (0, 0, 0) if active_tool == "ERASER" else draw_color
                current_size = eraser_thickness if active_tool == "ERASER" else brush_thickness
                cv2.circle(frame, (x1, y1), int(current_size / 2) + 2, current_color, cv2.FILLED)
                cv2.circle(frame, (x1, y1), int(current_size / 2) + 4, (0, 255, 0), 2)

                if not pinch_was_closed or (xp == 0 and yp == 0):
                    xp, yp = x1, y1
                else:
                    dist_step = math.hypot(x1 - xp, y1 - yp)
                    if dist_step > 2.0:
                        if active_tool == "ERASER":
                            cv2.line(math_canvas, (xp, yp), (x1, y1), (0, 0, 0), current_size, cv2.LINE_AA)
                            cv2.line(art_canvas, (xp, yp), (x1, y1), (0, 0, 0), current_size, cv2.LINE_AA)
                        else:
                            cv2.line(get_target_canvas(), (xp, yp), (x1, y1), current_color, current_size, cv2.LINE_AA)
                        xp, yp = x1, y1

                mode_tag = "ERASING" if active_tool == "ERASER" else f"DRAW ({active_layer})"
                cv2.putText(frame, mode_tag, (x1 + 15, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            else:
                xp, yp = 0, 0
                cv2.circle(frame, (x1, y1), 8, (0, 255, 255), 2)

            pinch_was_closed = pinch_now

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17),
            ]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start, end in connections:
                if 0 <= start < len(points) and 0 <= end < len(points):
                    cv2.line(frame, points[start], points[end], (200, 200, 200), 1)

            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(frame, points[4], 5, (0, 255, 255), -1)

        else:
            if shape_start_pt is not None:
                shape_start_pt = None
            xp, yp = 0, 0
            smooth_x, smooth_y = 0.0, 0.0
            pinch_was_closed = False
            stroke_snapshot_saved = False

        if should_quit:
            break

        combined_ink = cv2.bitwise_or(get_combined_canvas(), preview_layer)
        img_gray = cv2.cvtColor(combined_ink, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, combined_ink)

        cv2.putText(
            frame,
            f"Tool: {active_tool} [{active_layer}] | S: Solve+Steps+Graph | G: Graph | L: Layer | M: Solid/Outline | Z: Undo | Y: Redo | C: Clear | Q: Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            2,
        )

        draw_math_panel(frame)

        # Render any graph queued by the background solver on the main thread
        while not pending_graph_queue.empty():
            try:
                graph_payload = pending_graph_queue.get_nowait()
                show_graph_window(graph_payload)
            except queue.Empty:
                break
            except Exception as graph_err:
                print("[Graph Plotter Error]", graph_err)
                show_math_message(["Graph error:", str(graph_err)])

        cv2.imshow("AirSight : Live Vision Canvas", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("s"), ord("S")):
            trigger_solve_steps_graph()

        elif key in (ord("g"), ord("G")):
            trigger_graph()

        elif key in (ord("l"), ord("L")):
            active_layer = "ART" if active_layer == "MATH" else "MATH"

        elif key in (ord("m"), ord("M")):
            shape_filled = not shape_filled

        elif key in (ord("z"), ord("Z")):
            perform_undo()

        elif key in (ord("y"), ord("Y")):
            perform_redo()

        elif key in (ord("c"), ord("C")):
            clear_canvas()
            xp, yp = 0, 0

        elif key in (ord("q"), ord("Q")):
            break

finally:
    cap.release()
    hand_detector.close()
    cv2.destroyAllWindows()
