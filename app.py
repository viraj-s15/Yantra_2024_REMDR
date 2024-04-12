import tkinter as tk
from threading import Thread
from video_object_detection import run_detect_objects
import os
class Application:
    def __init__(self, master):
        self.master = master
        self.is_running = True
        master.title("Object Detection")

        self.run_time_label = tk.Label(master, text="Run Time:")
        self.run_time_label.pack()

        self.run_time_entry = tk.Entry(master)
        self.run_time_entry.pack()

        self.run_button = tk.Button(master, text="Run Detection", command=self.run_detection)
        self.run_button.pack()

        self.exit_button = tk.Button(master, text="Exit", command=self.exit_app)
        self.exit_button.pack()

    def run_detection(self):
        run_time = int(self.run_time_entry.get())
        thread = Thread(target=run_detect_objects, args=(run_time, self))
        thread.start()

    def exit_app(self):
        self.is_running = False
        self.master.destroy()
        os._exit(0)
        
root = tk.Tk()
app = Application(root)
root.mainloop()