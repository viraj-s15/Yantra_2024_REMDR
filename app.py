import customtkinter as ctk
from threading import Thread
from main import run_detection_in_app
import os

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Application:
    def __init__(self, master):
        self.master = master
        self.is_running = True
        master.title("REMDR")
        master.configure(bg='lightgrey')

        # Create three frames for the three columns
        self.left_frame = ctk.CTkFrame(master)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)

        self.center_frame = ctk.CTkFrame(master)
        self.center_frame.grid(row=0, column=1, padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(master)
        self.right_frame.grid(row=0, column=2, padx=10, pady=10)

        # Add the widgets to the left frame
        self.weights_label = ctk.CTkLabel(self.left_frame, text="Weights:")
        self.weights_label.grid(row=0, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.weights_entry = ctk.CTkEntry(self.left_frame)
        self.weights_entry.grid(row=0, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.source_label = ctk.CTkLabel(self.left_frame, text="Source:")
        self.source_label.grid(row=1, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.source_entry = ctk.CTkEntry(self.left_frame)
        self.source_entry.grid(row=1, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.data_label = ctk.CTkLabel(self.left_frame, text="Data:")
        self.data_label.grid(row=2, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.data_entry = ctk.CTkEntry(self.left_frame)
        self.data_entry.grid(row=2, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.imgsz_label = ctk.CTkLabel(self.left_frame, text="Image Size:")
        self.imgsz_label.grid(row=3, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.imgsz_entry = ctk.CTkEntry(self.left_frame)
        self.imgsz_entry.grid(row=3, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.conf_thres_label = ctk.CTkLabel(self.left_frame, text="Confidence Threshold:")
        self.conf_thres_label.grid(row=4, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.conf_thres_entry = ctk.CTkEntry(self.left_frame)
        self.conf_thres_entry.grid(row=4, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.iou_thres_label = ctk.CTkLabel(self.left_frame, text="NMS IOU Threshold:")
        self.iou_thres_label.grid(row=5, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.iou_thres_entry = ctk.CTkEntry(self.left_frame)
        self.iou_thres_entry.grid(row=5, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.max_det_label = ctk.CTkLabel(self.left_frame, text="Max Detections:")
        self.max_det_label.grid(row=6, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.max_det_entry = ctk.CTkEntry(self.left_frame)
        self.max_det_entry.grid(row=6, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.device_label = ctk.CTkLabel(self.left_frame, text="Device:")
        self.device_label.grid(row=7, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.device_entry = ctk.CTkEntry(self.left_frame)
        self.device_entry.grid(row=7, column=1, sticky='nsew', padx=(10, 100), pady=15)

        

        self.view_img_label = ctk.CTkLabel(self.left_frame, text="Show Results:")
        self.view_img_label.grid(row=8, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.view_img_entry = ctk.CTkEntry(self.left_frame)
        self.view_img_entry.grid(row=8, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.save_txt_label = ctk.CTkLabel(self.left_frame, text="Save Results to *.txt:")
        self.save_txt_label.grid(row=9, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.save_txt_entry = ctk.CTkEntry(self.left_frame)
        self.save_txt_entry.grid(row=9, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.save_conf_label = ctk.CTkLabel(self.left_frame, text="Save Confidences in --save-txt Labels:")
        self.save_conf_label.grid(row=10, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.save_conf_entry = ctk.CTkEntry(self.left_frame)
        self.save_conf_entry.grid(row=10, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.save_crop_label = ctk.CTkLabel(self.left_frame, text="Save Cropped Prediction Boxes:")
        self.save_crop_label.grid(row=11, column=0, sticky='nsew', padx=(40, 0), pady=15)
        self.save_crop_entry = ctk.CTkEntry(self.left_frame)
        self.save_crop_entry.grid(row=11, column=1, sticky='nsew', padx=(10, 100), pady=15)

        self.run_button = ctk.CTkButton(self.left_frame, text="Run Detection", command=self.run_detection, corner_radius=10)
        self.run_button.grid(row=12, column=0, columnspan=2, pady=10)

        self.exit_button = ctk.CTkButton(self.left_frame, text="Exit", command=self.exit_app, corner_radius=10)
        self.exit_button.grid(row=13, column=0, columnspan=2, pady=10)

        # Add placeholders to the center and right frames
        self.center_placeholder = ctk.CTkLabel(self.center_frame, text="Camera feed goes here")
        self.center_placeholder.pack()

        self.right_placeholder = ctk.CTkLabel(self.right_frame, text="Other stuff goes here")
        self.right_placeholder.pack()

    def run_detection(self):
        # Get the values from the ctk.CTkEntry widgets and pass them to the run_detection_in_app function
        weights = self.weights_entry.get()
        source = self.source_entry.get()

        # ... repeat for all other arguments ...
        args = (weights, source)
        run_detection_in_app(*args)

    def exit_app(self):
        os._exit(0)

if __name__ == "__main__":
    root = ctk.CTk()
    app = Application(root)
    root.mainloop()