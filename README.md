# Velo

Browser-based movement and form analysis using MediaPipe pose landmarks. Load a photo or video to inspect joint angles, angular velocity, hip motion, and force estimates.

## Run locally

```sh
python -m http.server 8000
```

Open http://localhost:8000. The page loads its pose model and MediaPipe runtime from external hosts; media analysis runs in the browser.

## Structure

- `index.html`: interface and analysis controls.
- `app.js`: pose detection, measurements, charts, and analysis state.
- `style.css`: layout and visual styling.
- `img/sample-squat.jpg`: example image.

Joint and force values are estimates from the detected pose and analysis assumptions. Camera angle, visibility, and landmark quality affect the measurements.
