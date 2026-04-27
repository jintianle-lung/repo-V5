# Standalone App Bundle

This folder packages a self-contained GUI runner for the released R5 model.

## Run

From the repository root:

```bash
python app_release/main.py
```

The app loads the released R5 detector/residual checkpoints from `checkpoints/` and `results/r5/`, then opens a local CSV playback interface.

## Demo Capture

To regenerate the preview image:

```bash
python app_release/capture_demo.py
```

## Notes

- The app is offline-first. It does not require a serial device to open.
- It uses the repository's bundled `data/raw/` files and released weights.
- The capture script writes a local runtime screenshot for verification.
