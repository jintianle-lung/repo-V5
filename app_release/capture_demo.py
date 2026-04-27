from __future__ import annotations

from pathlib import Path
import time
import tkinter as tk

from PIL import ImageGrab
import numpy as np

from standalone_gui import StandaloneApp, DEFAULT_SAMPLE_PATH


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "preview_runtime.png"


def main() -> None:
    root = tk.Tk()
    app = StandaloneApp(root, auto_load=False)
    app.load_csv(DEFAULT_SAMPLE_PATH, focus_frame=633)
    if app.data is not None:
        focus = int(np.argmax(app.data.sum(axis=1)))
        app.frame_scale.set(focus)
        app.current_frame = focus
        app.update_display()

    def capture_and_exit() -> None:
        root.update_idletasks()
        root.update()
        bbox = (
            root.winfo_rootx(),
            root.winfo_rooty(),
            root.winfo_rootx() + root.winfo_width(),
            root.winfo_rooty() + root.winfo_height(),
        )
        ImageGrab.grab(bbox=bbox).save(OUT_PATH)
        print(str(OUT_PATH))
        root.destroy()

    root.after(3500, capture_and_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
