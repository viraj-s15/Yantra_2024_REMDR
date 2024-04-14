from pynput import keyboard
import time
import os

def move_up():
    print("Moving up")

def move_down():
    print("Moving down")

def move_left():
    print("Moving left")

def move_right():
    print("Moving right")

def dance():
    print("Dancing")

key_to_function = {
    keyboard.Key.up: move_up,
    keyboard.Key.down: move_down,
    keyboard.Key.left: move_left,
    keyboard.Key.right: move_right,
    keyboard.Key.space: dance
}

def on_press(key):
    if key in key_to_function:
        key_to_function[key]()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()