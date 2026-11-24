"""
SMART HANDWRITTEN MATH MODULE (AI-Powered Vision + Calculus + Step-by-Step)
Used by air_canvas_smart_math_final_2.py

Recognition & solving flow:
1. Extract colored drawing strokes from the black Air Canvas.
2. Crop tightly and save to captured_equation.png.
3. Send image to Google Gemini Vision API (with automatic fallback on 429/503/504/timeout)
   to recognize regular equations OR calculus problems (derivatives, integrals,
   limits, ODEs) and generate steps.
4. Verify/compute with SymPy when applicable and return (latex_text, answer, steps).
5. Plot 2D equations/expressions in x and/or y (with roots or intersections),
   and skip 2D graphing for 3+ variable problems (such as systems with x, y, z).
"""

import json
import os
import re
import time
import cv2
import numpy as np
import sympy as sp
import matplotlib

# Ensure Matplotlib uses an interactive GUI backend instead of non-interactive Agg
# so pressing 'G' or auto-graphing opens a visible graph window alongside OpenCV.
for _backend in ("TkAgg", "QtAgg", "WXAgg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from latex2sympy2_extended import latex2sympy
from google import genai
from google.genai import types

AI_CLIENT = None
GRAPH_FIGURE = None
LAST_CAPTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "captured_equation.png",
)

# Ordered from fastest/lightweight models to heavier fallback models so 429/503/504
# automatically moves to the next available model.
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]


class Unsupported2DGraphError(ValueError):
    """Raised when an equation/system has 3+ variables or variables outside 2D (x, y)."""


def get_ai_client():
    """Initialize Gemini using the API key stored in .env."""
    global AI_CLIENT

    if AI_CLIENT is not None:
        return AI_CLIENT

    print("[Smart Math] Initializing Gemini AI Client...")

    try:
        from dotenv import load_dotenv

        project_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(project_dir, ".env")

        load_dotenv(dotenv_path=env_path, override=False)

        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key or not api_key.strip():
            raise RuntimeError(
                f"Gemini API key not found!\n"
                f"Expected .env file at: {env_path}\n"
                "Add GEMINI_API_KEY=YOUR_API_KEY to that file."
            )

        AI_CLIENT = genai.Client(api_key=api_key.strip())
        print("[Smart Math] Gemini AI client initialized successfully!")
        return AI_CLIENT

    except Exception as exc:
        AI_CLIENT = None
        raise RuntimeError(f"Gemini initialization failed: {exc}") from exc


def clean_latex_string(latex):
    """Clean OCR output, unwrap multi-line LaTeX blocks, and normalize operators."""
    if latex is None:
        return ""
    latex = str(latex).strip()
    latex = latex.replace("$", "").replace("\\,", " ")
    latex = latex.replace("\\!", "").replace("\\;", " ").replace("&", " ")

    latex = re.sub(r"\\equiv|\\approx|\\simeq|\\cong|\\to", "=", latex)
    latex = re.sub(r"```(?:latex|tex)?|```", "", latex).strip()
    latex = re.sub(r"\\(?:mathrm|mathbf|mathit|text)\{([^{}]*)\}", r"\1", latex)

    env_names = r"(?:cases|aligned|aligned\*|align|align\*|array|gather|gather\*|multline|multline\*)"
    latex = re.sub(r"\\left\s*\\\{", "", latex)
    latex = re.sub(r"\\right\s*(?:\\\}|\.)", "", latex)
    latex = re.sub(rf"\\begin\{{{env_names}\}}(?:\{{[^}}]*\}})?", "", latex)
    latex = re.sub(rf"\\end\{{{env_names}\}}", "", latex)

    latex = re.sub(r"[ \t]+", " ", latex)
    return latex.strip()


def _to_ascii_math(text):
    """Convert common math symbols to clean ASCII so OpenCV cv2.putText renders cleanly."""
    replacements = {
        "±": "+/-",
        "∫": "integral ",
        "∬": "double_integral ",
        "∂": "d",
        "√": "sqrt",
        "π": "pi",
        "θ": "theta",
        "∞": "inf",
        "→": " -> ",
        "⇒": " => ",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "²": "^2",
        "³": "^3",
        "⁺": "+",
        "⁻": "-",
    }
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out.encode("ascii", "replace").decode("ascii")


def _humanize_step_text(text):
    """
    Convert LaTeX tokens inside step/answer text into natural handwritten-style math
    (e.g., '\\pm' -> '+/-', '\\sqrt{4}' -> 'sqrt(4)', '\\frac{a}{b}' -> '(a)/(b)').
    """
    s = str(text).strip()
    s = re.sub(r"```(?:latex|tex|text)?|```", "", s).strip()
    s = s.replace("$", "")

    # Convert common LaTeX commands before general cleanup
    s = re.sub(r"\\pm\b", "+/-", s)
    s = re.sub(r"\\mp\b", "-/+", s)
    s = re.sub(r"\\times\b|\\cdot\b", "*", s)
    s = re.sub(r"\\div\b", "/", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:mathrm|mathbf|mathit|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\left\b|\\right\b", "", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\(\s*([^()]+?)\s*\)", lambda m: m.group(1) if re.fullmatch(r"[A-Za-z0-9_]+", m.group(1).strip()) else f"({m.group(1).strip()})", s)
    s = re.sub(r"[ \t]+", " ", s).strip()

    return _to_ascii_math(s)


def canvas_to_image(canvas):
    """
    Convert the BGR canvas (black background, colored strokes) into
    black handwriting on white, crop tightly, and save for the vision model.
    """
    if canvas is None or not isinstance(canvas, np.ndarray) or canvas.size == 0:
        raise ValueError("Canvas is unavailable. Draw an equation or calculus problem first.")

    if canvas.ndim == 2:
        color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    else:
        color = canvas

    strength = np.max(color, axis=2)
    foreground = (strength > 24).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    cleaned = np.zeros_like(foreground)
    for idx in range(1, count):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= 2:
            cleaned[labels == idx] = 255

    ys, xs = np.where(cleaned > 0)
    if len(xs) == 0:
        raise ValueError("No handwriting found. Pinch and draw before pressing S.")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    width, height = x2 - x1, y2 - y1
    margin = max(24, int(max(width, height) * 0.08))
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(cleaned.shape[1], x2 + margin), min(cleaned.shape[0], y2 + margin)

    ink = cleaned[y1:y2, x1:x2]
    prepared = 255 - ink

    h, w = prepared.shape
    target_h = 160
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    prepared = cv2.resize(prepared, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    prepared = cv2.copyMakeBorder(
        prepared, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255
    )

    cv2.imwrite(LAST_CAPTURE_PATH, prepared)
    return LAST_CAPTURE_PATH


def _should_try_next_model(err_str):
    """Return True when a model error is transient, rate-limited (429), or unavailable."""
    lower = err_str.lower()
    fallback_markers = (
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "503",
        "504",
        "unavailable",
        "deadline_exceeded",
        "10053",
        "404",
        "not_found",
        "timeout",
    )
    return any(marker in lower for marker in fallback_markers)


def _call_gemini_vision(client, image_bytes, prompt, schema=None):
    last_err = None
    for idx, model_name in enumerate(FALLBACK_MODELS):
        try:
            print(f"[Smart Math AI] Sending image to {model_name} ({idx + 1}/{len(FALLBACK_MODELS)})...")
            cfg_kwargs = {
                "http_options": types.HttpOptions(timeout=45000),
                "temperature": 0.1,
            }
            if schema is not None:
                cfg_kwargs["response_mime_type"] = "application/json"
                cfg_kwargs["response_schema"] = schema

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            print(f"[Smart Math AI] Response received from {model_name}.")
            return response.text
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            print(f"[Smart Math AI] {model_name} failed: {err_str}")
            if idx < len(FALLBACK_MODELS) - 1 and _should_try_next_model(err_str):
                print(f"[Smart Math AI] Switching to fallback model {FALLBACK_MODELS[idx + 1]}...")
                time.sleep(0.5)
                continue
            raise
    raise last_err


def _parse_latex_items(latex_text):
    """Split multi-equation LaTeX (separated by \\\\ or newlines) and parse each item."""
    parts = [p.strip() for p in re.split(r"\\\\|\r?\n", latex_text) if p.strip()]
    if not parts:
        parts = [latex_text]

    parsed = []
    for part in parts:
        cleaned_part = clean_latex_string(part)
        if not cleaned_part:
            continue
        parsed.append(latex2sympy(cleaned_part))
    return parsed


def recognize_equation(canvas, retry_graph_prompt=False):
    """Recognize handwritten equation/expression using Gemini across FALLBACK_MODELS."""
    image_path = canvas_to_image(canvas)
    client = get_ai_client()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    if retry_graph_prompt:
        prompt = (
            "Read the handwritten equation(s) in this image for 2D graphing. "
            "Return ONLY plain LaTeX equations in x and/or y (such as y = 2x or x + y = 3 \\\\ x - y = 1). "
            "Do not include markdown, units, or explanations."
        )
    else:
        prompt = (
            "Look at this image from an Air Canvas. Find all handwritten mathematical equation(s). "
            "Ignore stray background lines, artifacts, or hand jitter. "
            "Return ONLY plain LaTeX equations without \\begin{aligned}, \\begin{cases}, or & symbols. "
            "Separate multiple equations with \\\\ (for example: x + y = 3 \\\\ x - y = 1). "
            "Do not include markdown or explanations."
        )

    raw_latex = _call_gemini_vision(client, image_bytes, prompt)
    if raw_latex is None or not str(raw_latex).strip():
        raise ValueError("Gemini could not interpret the handwriting image.")

    latex_text = clean_latex_string(raw_latex)
    items = _parse_latex_items(latex_text)
    if not items:
        raise ValueError(f"Could not parse any mathematical expression from '{latex_text}'.")

    expression = items[0] if len(items) == 1 else items
    return latex_text, expression


def _sympy_verify_calculus(expr):
    """If SymPy parses a derivative, integral, or limit, return its exact string form."""
    try:
        if isinstance(expr, sp.Derivative):
            return sp.sstr(sp.simplify(expr.doit()))
        if isinstance(expr, sp.Integral):
            return sp.sstr(sp.simplify(expr.doit()))
        if isinstance(expr, sp.Limit):
            return sp.sstr(sp.simplify(expr.doit()))
    except Exception:
        pass
    return None


def _sympy_solve_equations(latex_text):
    """Attempt deterministic SymPy solving for regular equations or systems."""
    items = _parse_latex_items(latex_text)
    if not items:
        return None, None

    if isinstance(items, list) and len(items) > 1:
        if any(not isinstance(eq, sp.Equality) for eq in items):
            return None, None
        symbols = sorted(
            {sym for eq in items for sym in eq.free_symbols},
            key=lambda item: item.name,
        )
        if not symbols:
            return None, None
        sols = sp.solve(items, symbols, dict=True)
        if sols:
            formatted = ", ".join(
                f"{sym} = {sp.sstr(sp.simplify(sols[0][sym]))}"
                for sym in symbols
                if sym in sols[0]
            )
            steps = [
                f"System of {len(items)} equations in {', '.join(str(s) for s in symbols)}",
                f"Solved simultaneously -> {formatted}",
            ]
            return formatted, steps
        return None, None

    expr = items[0]
    calc_val = _sympy_verify_calculus(expr)
    if calc_val is not None:
        return calc_val, None

    if isinstance(expr, sp.Equality):
        symbols = sorted(expr.free_symbols, key=lambda item: item.name)
        if len(symbols) == 1:
            var = symbols[0]
            sols = sp.solve(expr, var)
            if sols:
                ans = ", ".join(f"{var} = {sp.sstr(sp.simplify(v))}" for v in sols)
                lhs_minus_rhs = sp.simplify(expr.lhs - expr.rhs)
                steps = [
                    f"Move all terms to one side: {sp.sstr(lhs_minus_rhs)} = 0",
                    f"Solve for {var}: {ans}",
                ]
                return ans, steps

    return None, None


def solve_handwriting(canvas):
    """
    Recognize handwritten algebra or calculus from canvas, solve with step-by-step
    explanation via Gemini + SymPy verification, and return:
        (latex_text, answer_text, steps_list)
    """
    image_path = canvas_to_image(canvas)
    client = get_ai_client()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "latex": {"type": "STRING"},
            "problem_type": {"type": "STRING"},
            "steps": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "final_answer": {"type": "STRING"},
        },
        "required": ["latex", "problem_type", "steps", "final_answer"],
    }

    prompt = (
        "You are a math tutor reading handwritten work from an Air Canvas. "
        "Identify the handwritten math problem, which may be a regular equation/system "
        "OR a calculus problem (derivative d/dx, indefinite/definite integral, limit, or ODE). "
        "Ignore stray background dots or hand jitter. "
        "Write steps and final_answer the way a person writes them by hand on notebook paper: "
        "use plain readable math like '+/-', 'sqrt(4)', '2', 'x = 2 or x = -2', 'x^2', 'sin(x)', "
        "and NEVER output LaTeX backslash commands like '\\pm', '\\sqrt', or '\\frac'. "
        "Return JSON with:\n"
        "1. 'latex': clean LaTeX of the recognized problem (use \\\\ between multiple equations).\n"
        "2. 'problem_type': one of 'derivative', 'integral', 'limit', 'ode', 'equation', 'system', 'expression'.\n"
        "3. 'steps': 2 to 5 concise, human-readable step strings showing how to solve it by hand.\n"
        "4. 'final_answer': the concise final answer in plain readable math (e.g. 'x = 2 or x = -2', '3*x^2 + 2', 'x^3/3 + C')."
    )

    raw_json = _call_gemini_vision(client, image_bytes, prompt, schema=response_schema)
    if not raw_json:
        raise ValueError("Gemini returned an empty response.")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse structured response from Gemini: {raw_json}") from exc

    latex_text = clean_latex_string(data.get("latex", ""))
    problem_type = str(data.get("problem_type", "")).strip().lower()
    raw_steps = data.get("steps") or []
    final_answer = _humanize_step_text(str(data.get("final_answer", "")).strip())

    steps = [_humanize_step_text(s) for s in raw_steps if str(s).strip()]

    try:
        if problem_type in ("equation", "system"):
            sympy_ans, sympy_steps = _sympy_solve_equations(latex_text)
            if sympy_ans:
                final_answer = _humanize_step_text(sympy_ans)
                if not steps and sympy_steps:
                    steps = [_humanize_step_text(st) for st in sympy_steps]
        elif problem_type in ("derivative", "integral", "limit"):
            items = _parse_latex_items(latex_text)
            if items:
                calc_val = _sympy_verify_calculus(items[0])
                if calc_val:
                    sympy_clean = _humanize_step_text(calc_val)
                    if problem_type == "integral" and "\\int" in latex_text and not re.search(r"\\int_[^^]+\^", latex_text):
                        if "+ C" not in final_answer and "+c" not in final_answer.lower():
                            final_answer = f"{sympy_clean} + C"
                    elif not final_answer:
                        final_answer = sympy_clean
    except Exception:
        pass

    if not final_answer:
        raise ValueError("Could not determine a final answer.")

    if not steps:
        steps = [f"Evaluated {problem_type or 'expression'}: {final_answer}"]

    return _to_ascii_math(latex_text), final_answer, steps


def _single_equation_to_plot_items(expression, x_symbol, y_symbol):
    """
    Convert a single SymPy equation or expression into:
      (y_series_list, vertical_x_list)
    - If y is present in an equality (e.g. y = 2x + 1, x^2 + y^2 = 9), solve for y.
    - If y is NOT present and it's a linear equation in x (e.g. x = 4, 2x - 6 = 0),
      treat it as a vertical line x = constant parallel to the y-axis.
    - If it's a non-linear single-variable equation (e.g. x^2 = 4) or a standalone
      expression (e.g. x^2 + 5x + 6), plot y = lhs - rhs (or y = expr) as a curve.
    """
    y_series = []
    vertical_lines = []

    if isinstance(expression, sp.Equality):
        equation = expression
        if y_symbol in equation.free_symbols:
            y_sols = sp.solve(equation, y_symbol)
            for y_expr in y_sols:
                y_simp = sp.simplify(y_expr)
                y_series.append((y_simp, f"y = {sp.sstr(y_simp)}"))
            return y_series, vertical_lines

        if x_symbol in equation.free_symbols:
            diff_expr = sp.simplify(equation.lhs - equation.rhs)
            poly = sp.Poly(diff_expr, x_symbol) if diff_expr.is_polynomial(x_symbol) else None
            if poly is not None and poly.degree() == 1:
                x_sols = sp.solve(equation, x_symbol)
                for x_expr in x_sols:
                    x_simp = sp.simplify(x_expr)
                    if not x_simp.has(x_symbol) and x_simp.is_real:
                        x_float = float(x_simp.evalf())
                        if np.isfinite(x_float):
                            vertical_lines.append((x_float, f"x = {x_float:g}"))
                return y_series, vertical_lines

            y_curve = diff_expr
            y_series.append((y_curve, f"y = {sp.sstr(y_curve)}"))
            return y_series, vertical_lines

        return y_series, vertical_lines

    fx = sp.simplify(expression)
    if fx.has(x_symbol):
        y_series.append((fx, f"y = {sp.sstr(fx)}"))
    return y_series, vertical_lines


def _build_plot_data_from_items(items):
    """
    Build plot series for 2D equations/expressions in x and/or y.
    Raises Unsupported2DGraphError if 3+ variables (such as x, y, z) are present,
    since multi-variable systems with z cannot be shown on a 2D graph.
    """
    if not isinstance(items, list):
        items = [items]

    all_symbols = sorted(
        {s for item in items for s in item.free_symbols},
        key=lambda sym: sym.name,
    )
    extra_vars = [s for s in all_symbols if s.name not in ("x", "y")]
    if extra_vars or len(all_symbols) > 2:
        var_names = ", ".join(str(s) for s in all_symbols)
        raise Unsupported2DGraphError(
            f"3D/multi-variable system ({var_names}) cannot be shown on a 2D graph."
        )

    x_candidates = [s for s in all_symbols if s.name == "x"]
    y_candidates = [s for s in all_symbols if s.name == "y"]

    x_symbol = x_candidates[0] if x_candidates else sp.Symbol("x")
    y_symbol = y_candidates[0] if y_candidates else sp.Symbol("y")

    series = []
    vertical_lines = []
    for idx, item in enumerate(items, start=1):
        sub_series, sub_vlines = _single_equation_to_plot_items(item, x_symbol, y_symbol)
        for y_expr, base_label in sub_series:
            label = f"Eq {idx}: {base_label}" if len(items) > 1 else base_label
            series.append((y_expr, label))
        for x_val, base_label in sub_vlines:
            label = f"Eq {idx}: {base_label}" if len(items) > 1 else base_label
            vertical_lines.append((x_val, label))

    if not series and not vertical_lines:
        raise ValueError("Could not convert the recognized equation(s) into plotable 2D curves.")

    real_roots = []
    intersection_points = []

    if len(items) == 1 and not y_candidates and not vertical_lines:
        single = items[0]
        fx = sp.simplify(single.lhs - single.rhs) if isinstance(single, sp.Equality) else sp.simplify(single)
        try:
            raw_roots = sp.solve(fx, x_symbol)
            for r in raw_roots:
                r_eval = sp.simplify(r)
                if r_eval.is_real:
                    r_float = float(r_eval.evalf())
                    if np.isfinite(r_float):
                        real_roots.append(r_float)
        except Exception:
            real_roots = []

    elif len(items) >= 2 and x_candidates and y_candidates:
        eq_items = [it for it in items if isinstance(it, sp.Equality)]
        if len(eq_items) >= 2:
            try:
                sols = sp.solve(eq_items, [x_symbol, y_symbol], dict=True)
                for sol in sols:
                    if x_symbol in sol and y_symbol in sol:
                        xv = sp.simplify(sol[x_symbol])
                        yv = sp.simplify(sol[y_symbol])
                        if xv.is_real and yv.is_real:
                            xf = float(xv.evalf())
                            yf = float(yv.evalf())
                            if np.isfinite(xf) and np.isfinite(yf):
                                intersection_points.append((xf, yf))
            except Exception:
                intersection_points = []

    return {
        "x_symbol": x_symbol,
        "series": series,
        "vertical_lines": vertical_lines,
        "real_roots": real_roots,
        "intersection_points": intersection_points,
    }


def build_graph_payload_from_latex(latex_text):
    """Build graph payload from an already-recognized LaTeX string without extra API calls."""
    items = _parse_latex_items(latex_text)
    if not items:
        raise ValueError(f"Could not parse any mathematical expression from '{latex_text}'.")

    plot_info = _build_plot_data_from_items(items)
    return {
        "latex_text": _to_ascii_math(latex_text),
        **plot_info,
    }


def prepare_graph_data(canvas):
    """Run OCR + equation/system parsing across fallback Gemini models."""
    try:
        latex_text, expression = recognize_equation(canvas, retry_graph_prompt=False)
    except Unsupported2DGraphError:
        raise
    except Exception as first_err:
        print(f"[Graph Plotter] First recognition pass failed ({first_err}); retrying with graph prompt...")
        latex_text, expression = recognize_equation(canvas, retry_graph_prompt=True)

    items = expression if isinstance(expression, list) else [expression]
    plot_info = _build_plot_data_from_items(items)
    return {
        "latex_text": _to_ascii_math(latex_text),
        **plot_info,
    }


def show_graph_window(graph_data):
    """Create and display the Matplotlib graph window (MUST be called on the main thread)."""
    global GRAPH_FIGURE
    if graph_data is None:
        return None

    print(f"[Graph Plotter] Matplotlib backend: {matplotlib.get_backend()}")

    latex_text = graph_data["latex_text"]
    x_symbol = graph_data["x_symbol"]
    series = graph_data.get("series", [])
    vertical_lines = graph_data.get("vertical_lines", [])
    real_roots = graph_data.get("real_roots", [])
    intersection_points = graph_data.get("intersection_points", [])

    if GRAPH_FIGURE is not None:
        plt.close(GRAPH_FIGURE)

    GRAPH_FIGURE, ax = plt.subplots(figsize=(9, 6))
    x_min, x_max = -10.0, 10.0

    focus_x = (
        list(real_roots)
        + [pt[0] for pt in intersection_points]
        + [vx for vx, _ in vertical_lines]
    )
    if focus_x:
        span = max(abs(min(focus_x)), abs(max(focus_x))) + 5.0
        bound = max(10.0, min(50.0, span))
        x_min, x_max = -bound, bound

    x_values = np.linspace(x_min, x_max, 1200)
    plotted = False

    for y_expression, label in series:
        function = sp.lambdify(x_symbol, y_expression, modules=["numpy"])
        with np.errstate(all="ignore"):
            y_values = np.asarray(function(x_values))
        if y_values.ndim == 0:
            y_values = np.full_like(x_values, float(y_values))
        y_values = np.real_if_close(y_values)
        if np.iscomplexobj(y_values):
            continue
        y_values = np.asarray(y_values, dtype=float)
        y_values[~np.isfinite(y_values)] = np.nan
        ax.plot(x_values, y_values, linewidth=2.2, label=label)
        plotted = True

    for x_val, label in vertical_lines:
        ax.axvline(x_val, color="tab:blue", linewidth=2.2, label=label)
        plotted = True

    if not plotted:
        plt.close(GRAPH_FIGURE)
        GRAPH_FIGURE = None
        raise ValueError("The recognized equation(s) could not be plotted as real-valued graphs.")

    for rx in real_roots:
        ax.scatter([rx], [0.0], color="red", zorder=5, s=55, label=f"Root x = {rx:g}")

    for idx, (ix, iy) in enumerate(intersection_points):
        pt_label = f"Intersection ({ix:g}, {iy:g})" if idx == 0 else None
        ax.scatter([ix], [iy], color="crimson", zorder=6, s=65, label=pt_label)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Graph of {latex_text}")
    ax.legend()
    ax.set_xlim(x_min, x_max)
    GRAPH_FIGURE.tight_layout()

    backend_lower = matplotlib.get_backend().lower()
    if backend_lower == "agg":
        fallback_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "plotted_graph.png",
        )
        GRAPH_FIGURE.savefig(fallback_path, dpi=150)
        print(f"[Graph Plotter] Warning: Agg backend active; saved graph to {fallback_path}")
    else:
        plt.show(block=False)
        plt.pause(0.15)
        try:
            mgr = GRAPH_FIGURE.canvas.manager
            if mgr is not None and hasattr(mgr, "window") and hasattr(mgr.window, "lift"):
                mgr.window.lift()
        except Exception:
            pass

    return latex_text


def graph_handwriting(canvas):
    data = prepare_graph_data(canvas)
    return show_graph_window(data)
