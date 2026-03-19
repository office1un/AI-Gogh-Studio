import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torchvision.transforms as transforms
import shutil
import webbrowser
import ctypes
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
MODEL_PATH = "models/gogh_style_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIN_W, WIN_H = 1280, 800 
PANEL_H = 100   
INFO_H = 120    
FOOT_H = 100    

# --- 1. Model Definition ---
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True), nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True), nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        )
    def forward(self, x): return self.model(x)

# --- 2. Style Conversion Engine ---
class StyleConverter:
    def __init__(self, model_path):
        self.model = TransformerNet()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()

    def process(self, frame):
        if frame is None: return None
        h_orig, w_orig = frame.shape[:2]
        proc_w, proc_h = 400, 300
        img = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s = cv2.convertScaleAbs(s, alpha=2.0, beta=20) 
        v = cv2.convertScaleAbs(v, alpha=1.2, beta=0)  
        img_pre = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        input_tensor = transform(img_pre).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_data = output_tensor.cpu().clamp(0, 255).data[0].numpy()
        output_data = output_data.transpose(1, 2, 0).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_data, cv2.COLOR_RGB2BGR)
        y_orig = cv2.cvtColor(cv2.resize(frame, (proc_w, proc_h)), cv2.COLOR_BGR2GRAY)
        y_ai = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
        y_final = cv2.addWeighted(y_ai, 0.85, y_orig, 0.15, 0)
        res_hsv = cv2.merge([h, s, y_final])
        final_bgr = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)
        return cv2.resize(final_bgr, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

# --- 3. Main Application ---
class GoghStudio:
    def __init__(self, converter):
        self.conv = converter
        self.mode = "CAMERA"
        self.cap = cv2.VideoCapture(0)
        self.converted_cap = None
        self.original_cap = None
        self.converted_img = None
        self.original_img = None
        self.is_converting = False
        self.win_name = "AI Gogh Studio"
        self.mouse_pos = (0, 0)
        self.clicked_btn = None
        self.temp_file = "temp_converted_vid.mp4"
        self.sig_rect = [0, 0, 0, 0]

        self.is_recording = False
        self.rec_writer = None
        
        self.slider_val = 0
        self.is_dragging = False
        self.v_fps = 0
        self.v_total_frames = 0
        self.v_duration_sec = 0

        self.btns = {
            "CAMERA": [420, 10, 570, 90],
            "VIDEO":  [580, 10, 730, 90],
            "IMAGE":  [740, 10, 890, 90],
            "SAVE":   [900, 10, 1050, 90],
            "QUIT":   [WIN_W - 105, 10, WIN_W - 5, 90]
        }

        self.ui_images = {}
        img_configs = {
            "TITLE":  ("./images/title.png", (400, 90)),
            "CAMERA": ("./images/camera.png", (150, 80)),
            "VIDEO":  ("./images/video.png", (150, 80)),
            "IMAGE":  ("./images/image.png", (150, 80)),
            "SAVE":   ("./images/save.png", (150, 80)),
            "QUIT":   ("./images/close.png", (100, 80))
        }
        
        for name, (path, size) in img_configs.items():
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.ui_images[name] = cv2.resize(img, size) if img is not None else np.zeros((size[1], size[0], 4), dtype=np.uint8)

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02}:{s:02}"

    def get_layout(self, frame, target_h):
        target_w = WIN_W // 2
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if frame is None: return canvas
        f_h, f_w = frame.shape[:2]
        scale = min(target_w / f_w, target_h / f_h)
        new_w, new_h = int(f_w * scale), int(f_h * scale)
        res = cv2.resize(frame, (new_w, new_h))
        y_off, x_off = (target_h - new_h) // 2, (target_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = res
        return canvas

    def draw_custom_slider(self, canvas, y_start):
        bar_w = WIN_W - 100
        bar_x = 50
        bar_y = y_start + 20
        cv2.line(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y), (150, 150, 150), 2)
        ratio = self.slider_val / max(1, self.v_total_frames - 1)
        curr_x = int(bar_x + bar_w * ratio)
        cv2.line(canvas, (bar_x, bar_y), (curr_x, bar_y), (255, 255, 255), 3)
        cv2.circle(canvas, (curr_x, bar_y), 10, (255, 255, 255), -1)
        curr_sec = self.slider_val / self.v_fps if self.v_fps > 0 else 0
        t_curr = self.format_time(curr_sec)
        t_total = self.format_time(self.v_duration_sec)
        cv2.putText(canvas, t_curr, (10, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, t_total, (WIN_W - 70, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_ui(self, canvas):
        panel = canvas[:PANEL_H, :]
        self.overlay_alpha(panel, self.ui_images["TITLE"], 0, 10, bg_black=True)
        cursor_hand = False
        for name, r in self.btns.items():
            bx1, by1, bx2, by2 = r
            btn_img = self.ui_images[name].copy()
            if bx1 <= self.mouse_pos[0] <= bx2 and by1 <= self.mouse_pos[1] <= by2:
                cursor_hand = True
                btn_img[:,:,:3] = cv2.add(btn_img[:,:,:3], 40)
            self.overlay_alpha(panel, btn_img, bx1, by1, bg_black=(name=="QUIT"))

        txt_x = 30
        base_y = PANEL_H + 40
        cv2.putText(canvas, "Vincent Willem van Gogh (30 March 1853 - 29 July 1890)", (txt_x, base_y), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "A Dutch Post-Impressionist painter who is among the most famous and influential figures in the", 
                    (txt_x, base_y + 35), cv2.FONT_HERSHEY_COMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "history of Western art. (Partly quoted from Wikipedia.)", 
                    (txt_x, base_y + 65), cv2.FONT_HERSHEY_COMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        sig_text = "This application was developed by Office1un."
        font_candidates = ["Times New Roman Bold.ttf", "timesbd.ttf", "DejaVuSerif-Bold.ttf", "georgiab.ttf", "Georgia Bold.ttf"]
        font_pil = None
        for f_name in font_candidates:
            try:
                font_pil = ImageFont.truetype(f_name, 20)
                break
            except: continue
        if font_pil is None: font_pil = ImageFont.load_default()

        try:
            left, top, right, bottom = font_pil.getbbox(sig_text)
            tw, th = right - left, bottom - top
        except: tw, th = 380, 20

        tx, ty_base = (WIN_W - tw) // 2, WIN_H - 55 
        self.sig_rect = [tx, ty_base, tx + tw, ty_base + th + 10]

        if self.mode == "CAMERA" and not self.is_converting:
            instruction_text = "Click SAVE to start recording, click again to stop."
            instr_x = (WIN_W - 450) // 2  
            cv2.putText(canvas, instruction_text, (instr_x, ty_base - 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        is_hover = (self.sig_rect[0] <= self.mouse_pos[0] <= self.sig_rect[2] and 
                    self.sig_rect[1] <= self.mouse_pos[1] <= self.sig_rect[3])
        
        if is_hover:
            cursor_hand = True
            text_rgb = (136, 136, 136) 
        else:
            text_rgb = (255, 255, 255) 

        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((tx, ty_base), sig_text, font=font_pil, fill=text_rgb)
        line_y = ty_base + th - 1 
        draw.line([(tx, line_y), (tx + tw, line_y)], fill=text_rgb, width=1)
        canvas[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        try:
            if cursor_hand: cv2.setWindowProperty(self.win_name, cv2.WND_PROP_CURSOR, cv2.CURSOR_HAND)
            else: cv2.setWindowProperty(self.win_name, cv2.WND_PROP_CURSOR, cv2.CURSOR_ARROW)
        except: pass

    def handle_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        bar_y_start = WIN_H - FOOT_H - 20
        
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, r in self.btns.items():
                if r[0] <= x <= r[2] and r[1] <= y <= r[3]: self.clicked_btn = name
            
            if self.mode == "VIDEO_PREVIEW" and bar_y_start <= y <= bar_y_start + 40:
                self.is_dragging = True
                self.seek_video(x)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            self.seek_video(x)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            if self.sig_rect[0] <= x <= self.sig_rect[2] and self.sig_rect[1] <= y <= self.sig_rect[3]:
                webbrowser.open("https://www.office1un.com")
            if self.clicked_btn:
                if self.clicked_btn == "QUIT": self.running = False
                elif self.clicked_btn == "CAMERA": 
                    self.cleanup_temp_video(); self.mode = "CAMERA"; self.cap = cv2.VideoCapture(0)
                elif self.clicked_btn == "VIDEO": self.convert_video_file()
                elif self.clicked_btn == "IMAGE":
                    root = tk.Tk(); root.withdraw(); p = filedialog.askopenfilename()
                    if p:
                        self.original_img = cv2.imread(p)
                        if self.original_img is not None:
                            self.converted_img = self.conv.process(self.original_img)
                            self.mode = "IMAGE_PREVIEW"
                elif self.clicked_btn == "SAVE": self.save_result()
            self.clicked_btn = None

    def seek_video(self, mouse_x):
        if self.original_cap is None: return
        bar_w, bar_x = WIN_W - 100, 50
        rel_x = np.clip(mouse_x - bar_x, 0, bar_w)
        self.slider_val = int((rel_x / bar_w) * (self.v_total_frames - 1))
        self.original_cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider_val)
        self.converted_cap.set(cv2.CAP_PROP_POS_FRAMES, self.slider_val)

    def convert_video_file(self):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("MP4 Video", "*.mp4")])
        if not path: return
        self.cleanup_temp_video()
        v_in = cv2.VideoCapture(path)
        self.v_fps = v_in.get(cv2.CAP_PROP_FPS)
        self.v_total_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))
        self.v_duration_sec = self.v_total_frames / self.v_fps
        w, h = int(v_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_out = cv2.VideoWriter(self.temp_file, cv2.VideoWriter_fourcc(*'mp4v'), self.v_fps, (w, h))

        self.is_converting = True
        idx = 0
        h_area = WIN_H - PANEL_H - INFO_H - FOOT_H
        start_y = PANEL_H + INFO_H
        while True:
            ret, f = v_in.read()
            if not ret: break
            v_out.write(self.conv.process(f))
            idx += 1
            display = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            display[start_y : start_y + h_area, :WIN_W//2] = self.get_layout(f, h_area)
            self.draw_ui(display)
            
            txt = f"Converting... {int((idx/self.v_total_frames)*100)}%"
            t_size = cv2.getTextSize(txt, 2, 1, 2)[0]
            tx = WIN_W // 2 + (WIN_W // 2 - t_size[0]) // 2
            ty = start_y + (h_area + t_size[1]) // 2
            cv2.putText(display, txt, (tx, ty), 2, 1, (255,255,255), 2)
            cv2.imshow(self.win_name, display); cv2.waitKey(1)

        v_in.release(); v_out.release()
        self.is_converting, self.mode = False, "VIDEO_PREVIEW"
        self.original_cap, self.converted_cap = cv2.VideoCapture(path), cv2.VideoCapture(self.temp_file)

    def save_result(self):
        if self.mode == "CAMERA":
            if not self.is_recording:

                root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".mp4",
                    filetypes=[("MP4 Video", "*.mp4")],
                    title="Start Real-time Recording"
                )
                if save_path:

                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    self.rec_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                    self.is_recording = True
            else:
                self.is_recording = False
                if self.rec_writer:
                    self.rec_writer.release()
                    self.rec_writer = None
                messagebox.showinfo("Success", "Recording stopped and saved.")
            return 

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) 

        if self.mode == "VIDEO_PREVIEW":
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4")],
                title="Save Processed Video"
            )
            if save_path:
                save_path = os.path.abspath(save_path)
                try:
                    if os.path.exists(self.temp_file):
                        shutil.copy2(self.temp_file, save_path)
                        messagebox.showinfo("Success", f"Video saved successfully at:\n{save_path}")
                    else:
                        messagebox.showerror("Error", "Temporary video file not found.")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save video:\n{e}")

        elif self.mode == "IMAGE_PREVIEW":
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                title="Save Processed Image"
            )
            if save_path:
                save_path = os.path.abspath(save_path)
                try:
                    rgb_img = cv2.cvtColor(self.converted_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    
                    pil_img.save(save_path)
                    
                    messagebox.showinfo("Success", f"Image saved successfully at:\n{save_path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def cleanup_temp_video(self, release_only=False):
        if self.converted_cap: self.converted_cap.release(); self.converted_cap = None
        if self.original_cap: self.original_cap.release(); self.original_cap = None
        if not release_only and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass

    def overlay_alpha(self, back, fore, x, y, bg_black=False):
        fh, fw = fore.shape[:2]
        roi = back[y:y+fh, x:x+fw]
        if bg_black: roi[:] = 0
        if fore.shape[2] == 4:
            alpha = fore[:, :, 3] / 255.0
            for c in range(3): roi[:, :, c] = (alpha * fore[:, :, c] + (1.0 - alpha) * roi[:, :, c])
        else:
            roi[:, :, :3] = fore[:, :, :3]

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)
        temp_root = tk.Tk()
        temp_root.withdraw()

        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, self.win_name)
            if hwnd:

                style = ctypes.windll.user32.GetWindowLongW(hwnd, -16) 
                
                style &= ~0x00010000
                style &= ~0x00040000

                ctypes.windll.user32.SetWindowLongW(hwnd, -16, style)
        except Exception as e:
            print(f"Could not disable maximize button: {e}")

        cv2.setMouseCallback(self.win_name, self.handle_mouse)
        cv2.setMouseCallback(self.win_name, self.handle_mouse)
        self.running = True
        while self.running:
            display = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            h_area = WIN_H - PANEL_H - INFO_H - FOOT_H
            start_y = PANEL_H + INFO_H

            if self.mode == "CAMERA":
                ret, frame = self.cap.read()
                if ret:
                    processed = self.conv.process(frame)
                    
                    if self.is_recording and self.rec_writer:
                        self.rec_writer.write(processed)
                        cv2.circle(display, (WIN_W - 70, PANEL_H + 105), 10, (0, 0, 255), -1)
                        cv2.putText(display, "REC", (WIN_W - 55, PANEL_H + 112), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    display[start_y : start_y + h_area, :WIN_W//2] = self.get_layout(frame, h_area)
                    display[start_y : start_y + h_area, WIN_W//2:] = self.get_layout(processed, h_area)
            
            elif self.mode == "VIDEO_PREVIEW":
                if self.original_cap is not None:
                    if not self.is_dragging:
                        self.slider_val = int(self.original_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret1, f1 = self.original_cap.read()
                    ret2, f2 = self.converted_cap.read()
                    if not ret1:
                        self.original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.converted_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    display[start_y : start_y + h_area, :WIN_W//2] = self.get_layout(f1, h_area)
                    display[start_y : start_y + h_area, WIN_W//2:] = self.get_layout(f2, h_area)
                    self.draw_custom_slider(display, start_y + h_area - 20)
                
            elif self.mode == "IMAGE_PREVIEW":
                display[start_y : start_y + h_area, :WIN_W//2] = self.get_layout(self.original_img, h_area)
                display[start_y : start_y + h_area, WIN_W//2:] = self.get_layout(self.converted_img, h_area)

            self.draw_ui(display)
            cv2.imshow(self.win_name, display)
            if cv2.waitKey(20) & 0xFF == ord('q'): break
            
        if self.rec_writer:
            self.rec_writer.release()
        self.cleanup_temp_video()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    GoghStudio(StyleConverter(MODEL_PATH)).run()