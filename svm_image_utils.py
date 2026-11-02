import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy.ndimage import uniform_filter


def select_samples(image, title="选择训练样本", excluded=()):
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    scale = min(1, (root.winfo_screenwidth() - 280) / image.width, (root.winfo_screenheight() - 160) / image.height)
    width, height = int(image.width * scale), int(image.height * scale)
    photo = ImageTk.PhotoImage(image.resize((width, height)), master=root)
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0)
    canvas.pack(side="left")
    canvas.create_image(0, 0, image=photo, anchor="nw")
    panel = tk.Frame(root, padx=20, pady=20)
    panel.pack(side="right", fill="y")
    tk.Label(panel, text="左键：河水（青色）\n右键：非河水（红色）\n每次点击选取一个像素", justify="left").pack(pady=10)
    status = tk.Label(panel, text="河水：0\n非河水：0", justify="left")
    status.pack(pady=10)
    samples = {}
    markers = {}
    confirmed = False

    def click(event, label):
        x, y = int(event.x * image.width / width), int(event.y * image.height / height)
        if (x, y) in excluded:
            return
        samples[x, y] = label
        canvas.delete(markers.get((x, y), "unused"))
        color = "cyan" if label == 1 else "red"
        markers[x, y] = canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill=color, outline="black")
        water = list(samples.values()).count(1)
        land = list(samples.values()).count(-1)
        status.config(text=f"河水：{water}\n非河水：{land}")
        button.config(state="normal" if water and land else "disabled")

    def confirm():
        nonlocal confirmed
        confirmed = True
        root.destroy()

    button = tk.Button(panel, text="确定", width=14, command=confirm, state="disabled")
    button.pack(pady=15)
    tk.Label(panel, text="两类都选好后点击确定\n直接关闭窗口则取消", justify="left").pack()
    canvas.bind("<Button-1>", lambda event: click(event, 1))
    canvas.bind("<Button-3>", lambda event: click(event, -1))
    root.mainloop()
    if confirmed:
        return np.array([(x, y, label) for (x, y), label in samples.items()])


def prepare_data(image, samples, window=21):
    rgb = np.asarray(image, dtype=float) / 255
    width = image.width
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    mean = uniform_filter(gray, size=window)
    texture = np.sqrt(np.maximum(uniform_filter(gray ** 2, size=window) - mean ** 2, 0))
    pixels = rgb.reshape(-1, 3)
    color = pixels / (pixels.sum(axis=1, keepdims=True) + 1e-6)
    features = np.column_stack((pixels, color, texture.ravel()))
    indices = samples[:, 1] * width + samples[:, 0]
    y = samples[:, 2].astype(float)
    X = features[indices]
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (features - mean) / std, (X - mean) / std, y


def save_samples(image, samples, path):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for x, y, label in samples:
        color = "cyan" if label == 1 else "red"
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")
    marked.save(path)


def save_compare(image, masks, path, titles=("Input", "Primal SVM", "Dual SVM")):
    size = 500
    canvas = Image.new("RGB", (size * 3, size + 30), "white")
    draw = ImageDraw.Draw(canvas)
    pictures = [image, Image.fromarray(masks[0]), Image.fromarray(masks[1])]
    for i, (title, picture) in enumerate(zip(titles, pictures)):
        draw.text((i * size + 10, 8), title, fill="black")
        canvas.paste(picture.resize((size, size), Image.Resampling.NEAREST), (i * size, 30))
    canvas.save(path)
