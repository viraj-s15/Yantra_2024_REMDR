import customtkinter as ctk
from threading import Thread
from video_object_detection import run_detect_objects
import os

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Application:
    def __init__(self, master):
        self.master = master
        self.is_running = True
        master.title("Object Detection")
        master.configure(bg='lightgrey')

        self.frame = ctk.CTkFrame(master)
        self.frame.pack(padx=10, pady=10)

        self.run_time_label = ctk.CTkLabel(self.frame, text="Run Time:")
        self.run_time_label.grid(row=0, column=0, sticky='e', padx=(100, 10), pady=5)

        self.run_time_entry = ctk.CTkEntry(self.frame)
        self.run_time_entry.grid(row=0, column=1, sticky='w', padx=(10, 100), pady=5)

        self.run_button = ctk.CTkButton(self.frame, text="Run Detection", command=self.run_detection, corner_radius=10)
        self.run_button.grid(row=1, column=0, columnspan=2, padx=(10, 100), pady=5)

        self.exit_button = ctk.CTkButton(self.frame, text="Exit", command=self.exit_app, corner_radius=10)
        self.exit_button.grid(row=2, column=0, columnspan=2, padx=(10, 100), pady=5)

    def run_detection(self):
        run_time = int(self.run_time_entry.get())
        thread = Thread(target=run_detect_objects, args=(run_time, self))
        thread.start()

    def exit_app(self):
        self.is_running = False
        self.master.destroy()
        os._exit(0)

root = ctk.CTk()
app = Application(root)
root.mainloop()