import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def show_third_page(original_img: Image.Image, output_img: Image.Image, back_callback):
    third_window = tk.Toplevel()
    third_window.title("Comparison Result")
    third_window.geometry("900x500")
    third_window.configure(bg="white")

    tk.Label(third_window, text="輸出結果比較", font=("Arial", 16, "bold"), bg="white").pack(pady=10)

    # 顯示原圖與處理後圖片
    image_frame = tk.Frame(third_window, bg="white")
    image_frame.pack(pady=20)

    original_img_resized = original_img.resize((300, 300))
    output_img_resized = output_img.resize((300, 300))

    original_img_tk = ImageTk.PhotoImage(original_img_resized)
    output_img_tk = ImageTk.PhotoImage(output_img_resized)

    original_label = tk.Label(image_frame, image=original_img_tk, bg="white")
    original_label.image = original_img_tk
    original_label.pack(side="left", padx=30)

    output_label = tk.Label(image_frame, image=output_img_tk, bg="white")
    output_label.image = output_img_tk
    output_label.pack(side="right", padx=30)

    # 儲存圖片功能
    def save_output():
        output_img.save("output.jpg")
        status_label.config(text="✅ 圖片已儲存為 output.jpg")

    # 控制區塊
    button_frame = tk.Frame(third_window, bg="white")
    button_frame.pack()

    ttk.Button(button_frame, text="Save Image", command=save_output).pack(side="left", padx=10)
    ttk.Button(button_frame, text="Try Again", command=lambda: [third_window.destroy(), back_callback()]).pack(side="left", padx=10)

    status_label = tk.Label(third_window, text="", font=("Arial", 10), fg="green", bg="white")
    status_label.pack(pady=10)

    third_window.mainloop()
