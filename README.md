# 🎯 Color Cap Detection on Tube Sheet (YOLO + ANN)

Detects color caps (Red, Blue, Yellow, Green) placed on a circular tube sheet using YOLOv8 and a custom-trained ANN.


Here their is th video in input folder named `All_color_caps_inside_the_tube_under_bright_Lights.mp4` is the input for the code.
The code then give output for every five seconds of color detected cap (only few images are outputed).

## 🧠 How It Works

1. **Tube Detection** – Uses coordinates from `tube_locations.txt` to locate each tube hole.
2. **Cap Detection** – YOLOv8 detects color caps placed over the tubes.
3. **Color Classification** – ANN classifies the cap into six categories:
   - Red
   - Green
   - Blue
   - Yellow
   - No Cap (White/Grey)

4. **Output** – Detected tubes are color-coded and saved in output folder.

---

## 🧰 Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Custom ANN (PyTorch)
- NumPy, Matplotlib

---

## 🛠️ Setup

```bash
git clone https://github.com/Mitesh-C/color-cap-detection.git
cd color-cap-detection

# Set up environment
pip install -r requirements.txt

# Run the script
python src/cap_detector.py
