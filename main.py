import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from svm_primal_image_seg import solve_primal
from svm_dual_image_seg import solve_dual
from svm_rbf_image_seg import solve_rbf, predict_rbf
from svm_image_utils import select_samples, prepare_data


def main():
    folder = Path(__file__).resolve().parent.parent
    output = folder / "output" / "outputend"
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(folder / "00000341.jpg").convert("RGB")
    C, gamma, window = 1.0, 0.1, 21
    samples = select_samples(image)
    features, X, y = prepare_data(image, samples, window)

    start = perf_counter()
    w1, b1 = solve_primal(X, y, C)
    train1 = perf_counter() - start
    start = perf_counter()
    mask1 = (features @ w1 + b1 >= 0).astype(np.uint8) * 255
    predict1 = perf_counter() - start

    start = perf_counter()
    w2, b2, a2 = solve_dual(X, y, C)
    train2 = perf_counter() - start
    start = perf_counter()
    mask2 = (features @ w2 + b2 >= 0).astype(np.uint8) * 255
    predict2 = perf_counter() - start

    start = perf_counter()
    a, b = solve_rbf(X, y, C, gamma)
    train3 = perf_counter() - start
    start = perf_counter()
    mask3 = predict_rbf(features, X, y, a, b, gamma)
    predict3 = perf_counter() - start

    names = ["原空间线性SVM", "对偶空间线性SVM", "RBF非线性SVM"]
    masks = [mask1, mask2, mask3]
    for i, name in enumerate(names):
        masks[i] = masks[i].reshape(image.height, image.width)
        # 先写临时文件再替换，避免直接截断正在被读取的旧文件。
        temp = output / (name + ".tmp.png")
        Image.fromarray(masks[i]).save(temp)
        temp.replace(output / (name + ".png"))

    times = [(train1, predict1), (train2, predict2), (train3, predict3)]
    temp = output / "算法耗时.tmp.csv"
    with open(temp, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["算法", "训练时间（s）", "整图预测时间（s）", "总时间（s）"])
        for name, (train, predict) in zip(names, times):
            writer.writerow([name, train, predict, train + predict])
    temp.replace(output / "算法耗时.csv")

    # 两个线性解代入同一个原问题，再与对偶目标比较。
    p1 = 0.5 * w1 @ w1 + C * np.maximum(0, 1 - y * (X @ w1 + b1)).sum()
    p2 = 0.5 * w2 @ w2 + C * np.maximum(0, 1 - y * (X @ w2 + b2)).sum()
    d = a2.sum() - 0.5 * w2 @ w2
    metrics = [
        ["原问题目标函数值（原问题法解）", p1],
        ["原问题目标函数值（对偶法解）", p2],
        ["对偶问题目标函数值（线性）", d],
        ["原始—对偶目标值差 P1-D", p1 - d],
        ["对偶法解的目标值差 P2-D", p2 - d],
        ["权重向量差异（L2范数）", np.linalg.norm(w1 - w2)],
        ["偏置差异（绝对值）", abs(b1 - b2)],
        ["线性分割一致率（%）", np.mean(mask1 == mask2) * 100],
        ["线性不同分类像素数量", np.sum(mask1 != mask2)],
        ["对偶等式约束误差", abs(a2 @ y)],
        ["对偶边界约束违反量", max(0, -a2.min(), a2.max() - C)],
        ["河水训练样本数", np.sum(y == 1)],
        ["非河水训练样本数", np.sum(y == -1)],
        ["整图像素数", image.width * image.height],
        ["C", C], ["gamma（RBF）", gamma], ["纹理窗口边长", window]
    ]
    temp = output / "线性SVM对比指标.tmp.csv"
    with open(temp, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["对比指标", "数值"])
        writer.writerows(metrics)
    temp.replace(output / "线性SVM对比指标.csv")


if __name__ == "__main__":
    main()
