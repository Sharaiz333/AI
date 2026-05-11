import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os

from knn_model import load_knn
from nn_model import load_model
print("*** RUNNING GUI ***\n")

CANVAS_SIZE  = 280
BRUSH_RADIUS = 13

BG_COLOR   = "#1a1a2e"
ACCENT     = "#e94560"
PANEL_BG   = "#16213e"
MUTED      = "#8888aa"
SUCCESS    = "#44dd88"
WARNING    = "#ffaa00"


# ===== LOAD MODELS =====
def load_models():
    try:
        knn = load_knn()
    except:
        knn = None

    try:
        nn = load_model()
    except:
        nn = None

    if knn is None and nn is None:
        messagebox.showerror(
            "Error",
            "No models found!\nRun train.py first."
        )

    return knn, nn


# ===== PREPROCESS =====
def preprocess_canvas(pil_image):
    img = pil_image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)

    if arr.max() == 0:
        return None

    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    digit_img = Image.fromarray(cropped.astype(np.uint8))
    digit_img = digit_img.resize((20, 20), Image.LANCZOS)

    canvas_28 = Image.new('L', (28, 28), 0)
    canvas_28.paste(digit_img, (4, 4))

    arr_28 = np.array(canvas_28) / 255.0
    return arr_28


# ===== MAIN APP =====
class DigitRecognizerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition made by Sharaiz Ahmed & Umair Waseem")
        self.root.geometry("850x650")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.knn_model, self.nn_model = load_models()

        self.pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self.build_ui()

        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonPress-1>", self.on_draw)

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG_COLOR, pady=10)
        header.pack(fill='x')

        tk.Label(header,
                 text="Handwritten Digit Recognition",
                 font=("Courier New", 15, "bold"),
                 bg=BG_COLOR, fg=ACCENT).pack()

        tk.Label(header,
                 text="Sharaiz Ahmed (57288)  |  Umair Waseem (44296)",
                 font=("Courier New", 9),
                 bg=BG_COLOR, fg=MUTED).pack()

        body = tk.Frame(self.root, bg=BG_COLOR)
        body.pack(padx=20, pady=(0, 10))

        # ===== LEFT =====
        left = tk.Frame(body, bg=BG_COLOR)
        left.pack(side='left', padx=(0, 16))

        tk.Label(left, text="Draw a digit (0–9):",
                 font=("Courier New", 10),
                 bg=BG_COLOR, fg="#cccccc").pack(pady=(0, 5))

        self.canvas = tk.Canvas(left,
                                width=CANVAS_SIZE,
                                height=CANVAS_SIZE,
                                bg="white",
                                cursor="crosshair",
                                highlightthickness=2,
                                highlightbackground=ACCENT)
        self.canvas.pack()

        tk.Label(left,
                 text="Hold left mouse button and draw",
                 font=("Courier New", 8),
                 bg=BG_COLOR,
                 fg="#555577").pack(pady=(4, 0))

        # ===== RIGHT PANEL =====
        panel = tk.Frame(body, bg=PANEL_BG, padx=18, pady=16, relief='ridge', bd=1)
        panel.pack(side='left', fill='y')

        tk.Label(panel, text="Result",
                 font=("Courier New", 11, "bold"),
                 bg=PANEL_BG, fg=MUTED).pack()

        self.result_label = tk.Label(panel, text="?",
                                     font=("Courier New", 78, "bold"),
                                     bg=PANEL_BG, fg=ACCENT, width=3)
        self.result_label.pack(pady=(4, 4))

        self.confidence_label = tk.Label(panel, text="Confidence:\n—",
                                         font=("Courier New", 10),
                                         bg=PANEL_BG, fg=MUTED,
                                         justify='center')
        self.confidence_label.pack(pady=(0, 14))

        self.status_label = tk.Label(panel,
                                     text="Draw a digit, then click Predict",
                                     font=("Courier New", 9),
                                     bg=PANEL_BG, fg=MUTED,
                                     justify='center')
        self.status_label.pack(pady=(0, 14))

        # ===== MODEL SELECT =====
        tk.Label(panel, text="Select Model:",
                 font=("Courier New", 9),
                 bg=PANEL_BG, fg=MUTED).pack()

        self.model_var = tk.StringVar(value="Neural Network")

        model_menu = tk.OptionMenu(panel, self.model_var,
                                  "Neural Network", "KNN")
        model_menu.config(bg="#2a2a4a", fg="white", relief='flat')
        model_menu.pack(fill='x', pady=(0, 12))

        # ===== BUTTONS =====
        tk.Button(panel, text="Predict",
                  font=("Courier New", 12, "bold"),
                  bg=ACCENT, fg="white",
                  relief='flat',
                  command=self.predict_digit).pack(fill='x', pady=(0, 7))

        tk.Button(panel, text="Clear",
                  font=("Courier New", 12),
                  bg="#2a2a4a", fg="#cccccc",
                  relief='flat',
                  command=self.clear_canvas).pack(fill='x')

        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg=BG_COLOR)
        footer.pack(fill='x', pady=(5, 10))

        tk.Label(footer, text="Artificial Intelligence Semester Project",
                 font=("Courier New", 10, "bold"),
                 bg=BG_COLOR, fg=ACCENT).pack()

        tk.Label(footer,
                 text="Neural Network | K-Nearest Neighbors (KNN)",
                 font=("Courier New", 9),
                 bg=BG_COLOR, fg=MUTED).pack()

        tk.Label(footer,
                 text="BSCS 6-1 — Riphah International University",
                 font=("Courier New", 9),
                 bg=BG_COLOR, fg=MUTED).pack()

        tk.Label(footer,
                 text="Submitted to: Mr. Junaid Khan",
                 font=("Courier New", 9),
                 bg=BG_COLOR, fg=MUTED).pack()

    # ===== DRAW =====
    def on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill="#111111", outline="#111111")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

    # ===== PREDICT =====
    def predict_digit(self):
        processed = preprocess_canvas(self.pil_image)
        
        if processed is None:
            self.status_label.config(text="Nothing drawn!", fg=WARNING)
            return
        
        if not is_valid_digit(processed):
            self.status_label.config(text="Invalid input! Draw a clear digit (0–9).", fg=WARNING)
            return

        model_choice = self.model_var.get()

        try:
            if model_choice == "Neural Network":
                data = processed.reshape(1, 28, 28, 1)
                pred = self.nn_model.predict(data, verbose=0)
                prediction = np.argmax(pred)
                confidence = pred[0][prediction] * 100
            else:
                data = processed.flatten().reshape(1, -1)
                prediction = self.knn_model.predict(data)[0]
                probas = self.knn_model.predict_proba(data)[0]
                confidence = probas[prediction] * 100

            self.result_label.config(text=str(prediction))
            self.confidence_label.config(
                text=f"Confidence:\n{confidence:.1f}%",
                fg=SUCCESS if confidence >= 75 else WARNING
            )

            if confidence < 75:
                self.status_label.config(
                    text="Invalid input! Not a clear digit.",
                    fg=WARNING
                )
                self.result_label.config(text="?")
                return

            self.status_label.config(
                text=f"{model_choice} Prediction: {prediction}",
                fg=SUCCESS
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ===== CLEAR =====
    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence:\n—", fg=MUTED)
        self.status_label.config(
            text="Draw a digit, then click Predict",
            fg=MUTED
        )


def is_valid_digit(arr):
    ink_ratio = np.sum(arr > 0.2) / arr.size

    if ink_ratio < 0.03:
        return False

    if ink_ratio > 0.5:
        return False

    coords = np.argwhere(arr > 0.2)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    height = y_max - y_min
    width  = x_max - x_min

    if height == 0 or width == 0:
        return False

    ratio = height / width
    if ratio > 4 or ratio < 0.25:
        return False

    return True


# ===== RUN =====
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()