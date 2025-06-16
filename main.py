import tkinter as tk
from tkinter import ttk
from second_page import create_second_page

def open_second_page():
    root.destroy()  # 關閉第一頁
    create_second_page()  # 開啟第二頁

# 第一頁主視窗
root = tk.Tk()
root.title("主畫面")
root.geometry("700x700")

# Title 標題
title_label = ttk.Label(
    root,
    text="Style-Guided Portrait Generation\nFrom Realistic Faces to Comic, 3D Rendered, and Beauty Styles",
    font=("Segoe UI Emoji", 16),
    justify="left"
)
title_label.pack(pady=10, anchor="w", padx=20)

# Instruction 標籤
instructions = """🧑‍💻 User Instructions

Welcome to the Style-Guided Portrait Generator!
This tool transforms your realistic facial photo into three distinct styles:
Beauty Filter, Western Cartoon, and 3D Rendered Cartoon.

🔹 Step 1: Load Your Photo
Click the “Load image” button to select a clear, front-facing photo of a human face.
Make sure the image is in JPG or PNG format.

🔹 Step 2: Choose a Style
Select one or more of the following styles:
💄 Beauty Filter
🎨 Comic (Western Cartoon)
🧊 3D Rendered Cartoon

🔹 Step 3: Generate Portrait
Click “Generate” to begin the transformation.
Please wait a few seconds while the system processes your image.

🔹 Step 4: View and Save Results
Your stylized portraits will appear on the result screen.
Click “Save” to save the image(s) or “Try Again” to re-load a new photo.
"""

instruction_label = tk.Label(
    root,
    text=instructions,
    font=("Segoe UI Emoji", 12),
    justify="left",
    anchor="nw",
    wraplength=650  # 控制換行寬度
)
instruction_label.pack(padx=20, pady=10, anchor="w")

# START 按鈕
start_button = ttk.Button(root, text="START", command=open_second_page)
start_button.pack(pady=20)

root.mainloop()
