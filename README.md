# Style-Guided-Portrait-Generation-From-Realistic-Faces-to-Comic-3D-Rendered-and-Beauty-Styles

A deep learning–based image generation system for transforming realistic facial photographs into multiple stylistic portraits, including **Beauty Filter**, **Comic (Western Cartoon)**, and **3D Rendered Cartoon** styles.

## 📌 Project Description

This project aims to implement a style-guided image generation system that accepts a real human face photo and automatically outputs stylized portraits in three distinct styles:

- **Beauty Filter Style**
- **Comic (Western Cartoon) Style**
- **3D Rendered Cartoon Style**

The system emphasizes not only the stylistic accuracy but also the preservation of essential facial identity and recognizable features. This aligns with modern challenges in **style-guided image generation**.

---

## 🎯 Objectives

- ✅ Build an automated image-to-image translation system for portrait stylization.
- ✅ Ensure the generated images are high-quality and stylistically accurate.
- ✅ Train deep learning models with effective style transformation capabilities.

---

## 🗂 Dataset

- 📁 The dataset contains **paired face images**: realistic input photos and their stylized versions.
- 🖼️ Stylized outputs are provided in three styles: Beauty, Comic, and 3D Cartoon.
- 📥 Download from the provided Google Drive link:  
  [Dataset Folder](https://drive.google.com/drive/folders/1k6g0WWvgzEhzJdJMv_ITE2j9f7CDMURJ?usp=drive_link)

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- torchvision  
- PIL (Python Imaging Library)  
- Tkinter (for GUI)

---

## 🚀 Installation

```bash
# Clone the repo
git clone https://github.com/zhencheng0410/Style-Guided-Portrait-Generation-From-Realistic-Faces-to-Comic-3D-Rendered-and-Beauty-Styles.git
cd Style-Guided-Portrait-Generation-From-Realistic-Faces-to-Comic-3D-Rendered-and-Beauty-Styles
```


## 🖥️ Usage

```bash
python main.py
```
1. Upload a realistic face image.

2. Choose a target style (Beauty / Comic / 3D).

3. The system will automatically generate the styled portrait.

4. View the result and optionally save or retry.