# AirSight: Live Vision Canvas & Smart Math Solver

**AirSight** is a touchless computer-vision whiteboard and AI-powered mathematics assistant built with Python, OpenCV, MediaPipe, SymPy, Matplotlib, and the Google Gemini Vision API. Draw equations, calculus problems, or diagrams in mid-air using hand pinch gestures, solve them step by step, and plot 2D graphs automatically.

---

## Key Features

- **Touchless Pinch-Controlled UI:** Select tools, colors, brush thicknesses, shape modes, undo/redo, clear, solve, graph, and **QUIT** directly in mid-air using index-thumb pinch gestures.
- **Dual-Layer Canvas (`MATH OCR` vs. `ART ONLY`):** Keep mathematical equations isolated on the `MATH` layer for clean recognition while sketching geometric shapes (`LINE`, `RECT`, `CIRCLE`, `ELLIPSE`), flood-filling (`FILL`), or sampling real-world colors (`EYEDROPPER`) on the `ART` layer.
- **Smart Algebra & Calculus Solver (`SOLVE+GRAPH`):**
  - Recognizes handwritten algebra, systems of equations, derivatives, integrals, and limits.
  - Displays human-readable, step-by-step solutions directly on the live camera HUD.
  - Verifies results symbolically with **SymPy** and includes automatic retry/backoff across fallback Gemini models.
- **2D Graph Plotter (`GRAPH`):**
  - Plots functions, parabolas (e.g., $x^2 + 5x + 6$, $x^2 = 4$), and linear equations in $x$ and $y$.
  - Accurately plots vertical lines parallel to the y-axis for constant equations (e.g., $x = 4$).
  - Highlights real roots and intersection points in an interactive Matplotlib window.

---

## Project Structure

```text
├── acsm_f.py              # Main live camera loop, MediaPipe hand tracking, UI toolbar, and HUD
├── sm_f.py                # Image preprocessing, Gemini Vision OCR/solver, SymPy verification, and Matplotlib plotter
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes .env, virtual environments, captured images, and build files
└── README.md              # Project documentation