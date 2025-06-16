import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from third_page import show_third_page
from beauty_unet_model import UNet as beauty_unet
from comic_unet_model import UNet as comic_unet
from threeD_unet_model import UNet as threeD_unet


# 模型載入
beauty_model = beauty_unet()
beauty_model.load_state_dict(torch.load("beauty_unet_model.pth", map_location="cpu", weights_only=True))
beauty_model.eval()

comic_model = comic_unet()
comic_model.load_state_dict(torch.load("comic_unet_model.pth", map_location="cpu", weights_only=True))
comic_model.eval()

threeD_model = threeD_unet()
threeD_model.load_state_dict(torch.load("styled_unet_model.pth", map_location="cpu", weights_only=True))
threeD_model.eval()

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 192)),
    transforms.ToTensor()
])

def create_second_page():
    uploaded_image_path = [None]
    window = tk.Tk()
    window.title("Generate Portrait")
    window.geometry("700x700")
    window.configure(bg="white")

    # === 左右區塊分隔 ===
    left_frame = tk.Frame(window, bg="white")
    left_frame.pack(side="left", fill="y", padx=20, pady=20)

    right_frame = tk.Frame(window, bg="white")
    right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    # === 左側：功能 ===

    # 上傳按鈕
    ttk.Button(left_frame, text="Upload Image", command=lambda: upload_image()).pack(anchor="w", pady=10)

    # 單選風格
    tk.Label(left_frame, text="Select One Style:", font=("Arial", 12, "bold"), bg="white").pack(anchor="w", pady=10)

    style_var = tk.StringVar(value="")
    ttk.Radiobutton(left_frame, text="💄 Beauty Filter", value="beauty", variable=style_var).pack(anchor="w", padx=10)
    ttk.Radiobutton(left_frame, text="🎨 Comic Style", value="comic", variable=style_var).pack(anchor="w", padx=10)
    ttk.Radiobutton(left_frame, text="🧊 3D Rendered Cartoon", value="3d", variable=style_var).pack(anchor="w", padx=10)

    # 產生按鈕
    ttk.Button(left_frame, text="Generate", command=lambda: generate_image()).pack(anchor="w", pady=20)

    # # 下載與重試
    # ttk.Button(left_frame, text="Download", command=lambda: print("Download")).pack(anchor="w", pady=5)
    # ttk.Button(left_frame, text="Try Again", command=window.destroy).pack(anchor="w", pady=5)

    # === 右側：預覽圖與狀態 ===
    preview_title = tk.Label(right_frame, text="預覽圖片", font=("Arial", 14, "bold"), bg="white", anchor="w", justify="left")
    preview_title.pack(anchor="w", padx=10)

    image_label = tk.Label(right_frame, bg="white")
    image_label.pack(anchor="w", padx=10, pady=(5, 10))

    status_label = tk.Label(right_frame, text="", font=("Arial", 10), fg="blue", bg="white", justify="left", anchor="w")
    status_label.pack(anchor="w", padx=10)

    tip_label = tk.Label(
        right_frame,
        text="📌 請確認圖片為正面人臉，解析度清晰。",
        font=("Arial", 10),
        fg="gray",
        bg="white",
        anchor="w",
        justify="left"
    )
    tip_label.pack(anchor="w", padx=10, pady=(5, 0))

    # === 功能函式 ===
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            uploaded_image_path[0] = file_path  # 儲存圖片路徑
            img = Image.open(file_path).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            status_label.config(text=f"Uploaded: {file_path.split('/')[-1]}")

    def generate_image():
        selected_style = style_var.get()
        if not uploaded_image_path[0]:
            status_label.config(text="請先上傳圖片")
            return

        if selected_style == "beauty":
            status_label.config(text="Generating style: beauty...")
            try:
                img = Image.open(uploaded_image_path[0]).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output_tensor = beauty_model(input_tensor)[0]

                output_image = transforms.ToPILImage()(output_tensor)

                # 呼叫第三頁
                def reopen_second_page():
                    window.destroy()
                    create_second_page()  # 重新載入第二頁

                show_third_page(original_img=img, output_img=output_image, back_callback=reopen_second_page)

            except Exception as e:
                status_label.config(text=f"⚠️ 錯誤: {str(e)}")

        elif selected_style == "comic":
            status_label.config(text="Generating style: comic...")
            try:
                # 載入圖片並處理
                img = Image.open(uploaded_image_path[0]).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output_tensor = comic_model(input_tensor)[0]
                output_tensor = output_tensor.clamp(-1, 1)
                output_tensor = (output_tensor + 1) / 2  # Tanh 輸出轉為 [0, 1]

                output_image = transforms.ToPILImage()(output_tensor)

                # 呼叫第三頁
                def reopen_second_page():
                    window.destroy()
                    create_second_page()

                show_third_page(original_img=img, output_img=output_image, back_callback=reopen_second_page)

            except Exception as e:
                status_label.config(text=f"⚠️ 錯誤: {str(e)}")

        elif selected_style == "3d":
            status_label.config(text="Generating style: 3D...")
            try:
                # 載入圖片並處理
                img = Image.open(uploaded_image_path[0]).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    style_id = torch.tensor([2], dtype=torch.long, device=input_tensor.device)
                    output = threeD_model(input_tensor, style_id)
                output_image = transforms.ToPILImage()(output.squeeze(0))  # 直接轉成 PIL.Image
                    # output_tensor = F.interpolate(output, size=size[::-1], mode='bilinear', align_corners=False)

                # output_image = transforms.ToPILImage()(output_tensor)

                # 呼叫第三頁
                def reopen_second_page():
                    window.destroy()
                    create_second_page()

                show_third_page(original_img=img, output_img=output_image, back_callback=reopen_second_page)

            except Exception as e:
                status_label.config(text=f"⚠️ 錯誤: {str(e)}")

        elif selected_style:
            status_label.config(text=f"{selected_style} 尚未實作")
        else:
            status_label.config(text="Please select a style before generating.")

    window.mainloop()
