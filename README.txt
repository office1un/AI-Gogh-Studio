============================================================
                AI Gogh Studio - User Guide
============================================================

[Overview]
AI Gogh Studio is a high-performance style transfer engine 
developed by Office1un. This application utilizes a Deep 
Learning model (TransformerNet) to transform camera feeds, 
videos, and static images into the unique artistic style 
of Vincent van Gogh in real-time.

[System Specifications]
- OS: Windows 10/11 (64-bit)
- GPU: NVIDIA GeForce RTX 3070 or higher (Recommended)
- CUDA Toolkit: 11.8
- Python: 3.10.x (Recommended)

[Installation & Setup]
1. Install NVIDIA CUDA Toolkit 11.8.
2. Ensure Python is installed and added to your system PATH.
3. Install dependencies using the provided manifest:
   
   pip install -r requirements.txt

   (Note: This will install PyTorch 2.1.0+cu118 and related 
   libraries optimized for GPU acceleration.)

4. Verify your environment:
   python -c "import torch; print(f'CUDA Active: {torch.cuda.is_available()}')"

[Project Structure]
- run.py                   : Main application executable
- ./models/                : Model asset directory
    - gogh_style_model.pth : Pre-trained neural network model
- requirements.txt         : Dependency list
- README.txt               : This user guide
- ./images/                : UI Assets directory
    - title.png            : Application branding
    - camera.png           : Camera mode icon
    - video.png            : Video conversion icon
    - image.png            : Image processing icon
    - save.png             : Export function icon
    - close.png            : Exit application icon

[How to Use]
1. Launch the application:
   python run.py

2. Select Mode:
   - CAMERA: Activate real-time transformation via webcam.
   - VIDEO : Select an MP4 file. A "Converting..." progress 
             indicator will appear during processing.
   - IMAGE : Select a static photo for artistic transformation.

3. Exporting:
   - Click the SAVE icon to export converted media.
   - For videos, conversion must complete before saving.

4. Exit:
   - Click the CLOSE (X) icon to safely close the studio.

[Interface Notes]
- The UI features a pure black background to enhance visual focus.
- A formal cursive signature "Office1un" is located at the 
  bottom center of the window.

[License]
MIT License

Copyright (c) 2026 Office1un

This project is licensed under the MIT License - see the LICENSE file for details.

------------------------------------------------------------
Developed by Shigenobu Anbo @ Office1un.
Kazuno City, Akita, Japan.
============================================================