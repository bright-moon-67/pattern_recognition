import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy.ndimage import uniform_filter


def sample_boxes(labels, boxes, count):
    rng = np.random.default_rng(0)
    samples = {}
    for x0, y0, x1, y1, label in boxes:
        rows, cols = np.nonzero(labels[y0:y1, x0:x1] == label)
        chosen = rng.choice(len(rows), min(count, len(rows)), replace=False)
        for i in chosen:
            samples[cols[i] + x0, rows[i] + y0] = label
    return np.array([(x, y, label) for (x, y), label in samples.items()], dtype=int).reshape(-1, 3)


def select_samples(image, title="框选训练样本", excluded=(), count=40):
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
    tk.Label(panel, text=f"左键拖动：框选河水\n右键拖动：框选非河水\n每个框最多抽取 {count} 个像素\n圆点表示实际抽中的像素", justify="left").pack(pady=10)
    status = tk.Label(panel, text="河水：0\n非河水：0", justify="left")
    status.pack(pady=10)
    labels = np.zeros((image.height, image.width), dtype=np.int8)
    boxes = []
    start = [0, 0]
    confirmed = False

    def press(event):
        start[:] = [event.x, event.y]
        canvas.delete("preview")
        color = "cyan" if event.num == 1 else "red"
        canvas.create_rectangle(*start, *start, outline=color, width=2, tags="preview")

    def drag(event):
        x, y = min(max(event.x, 0), width), min(max(event.y, 0), height)
        canvas.coords("preview", start[0], start[1], x, y)

    def release(event):
        drag(event)
        x0, y0, x1, y1 = canvas.coords("preview")
        x0, x1 = sorted([round(x0 * image.width / width), round(x1 * image.width / width)])
        y0, y1 = sorted([round(y0 * image.height / height), round(y1 * image.height / height)])
        if x0 == x1 or y0 == y1:
            canvas.delete("preview")
            return
        label = 1 if event.num == 1 else -1
        boxes.append((x0, y0, x1, y1, label))
        labels[y0:y1, x0:x1] = label
        for x, y in excluded:
            labels[y, x] = 0
        canvas.dtag("preview", "preview")
        samples = sample_boxes(labels, boxes, count)
        canvas.delete("sample")
        for x, y, label in samples:
            x, y = x * width / image.width, y * height / image.height
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="cyan" if label == 1 else "red", outline="black", tags="sample")
        water = np.sum(samples[:, 2] == 1)
        land = np.sum(samples[:, 2] == -1)
        status.config(text=f"河水样本：{water}\n非河水样本：{land}")
        button.config(state="normal" if water and land else "disabled")

    def confirm():
        nonlocal confirmed
        confirmed = True
        root.destroy()

    button = tk.Button(panel, text="确定", width=14, command=confirm, state="disabled")
    button.pack(pady=15)
    tk.Label(panel, text="两类都选好后点击确定\n直接关闭窗口则取消", justify="left").pack()
    for key in [1, 3]:
        canvas.bind(f"<ButtonPress-{key}>", press)
        canvas.bind(f"<B{key}-Motion>", drag)
        canvas.bind(f"<ButtonRelease-{key}>", release)
    root.mainloop()
    if confirmed:
        return sample_boxes(labels, boxes, count)


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
