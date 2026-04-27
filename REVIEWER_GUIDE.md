# Reviewer Guide

Start from the release directory:

```bash
cd github_reviewer_release
pip install -r requirements.txt
```

Quick checks:

```bash
python -m compileall tactile_inversion
python -m unittest discover tests
python -m tactile_inversion.demo --device cpu --no-scorecam
python -m tactile_inversion.evaluate --device cpu
```

Generate the task-guided Score-CAM figure:

```bash
python -m tactile_inversion.make_scorecam --device cpu
```

The released reference run is `results/r5`. The frozen detector run is `results/frozen_detector`.

