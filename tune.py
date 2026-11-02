import csv
from pathlib import Path

import numpy as np
from PIL import Image

from svm_image_utils import select_samples, prepare_data, save_samples, save_compare
from svm_primal_image_seg import solve_primal


def main():
    folder = Path(__file__).resolve().parent.parent
    output = folder / "output"
    output.mkdir(exist_ok=True)
    image = Image.open(folder / "00000341.jpg").convert("RGB")
    samples = select_samples(image, "调参：选择训练样本")
    if samples is None:
        return
    excluded = set(map(tuple, samples[:, :2]))
    validation = select_samples(image, "调参：在其他区域选择验证样本", excluded)
    if validation is None:
        return
    indices = validation[:, 1] * image.width + validation[:, 0]
    labels = validation[:, 2]
    save_samples(image, samples, output / "tuning_samples.png")
    save_samples(image, validation, output / "validation_regions.png")
    np.savetxt(output / "tuning_samples.csv", samples, fmt="%d", delimiter=",", header="x,y,label", comments="")
    np.savetxt(output / "validation_samples.csv", validation, fmt="%d", delimiter=",", header="x,y,label", comments="")
    records = []
    best_score = -1
    for window in [9, 15, 21]:
        features, X, y = prepare_data(image, samples, window)
        for C in [0.1, 1, 10]:
            w, b = solve_primal(X, y, C)
            pred = np.where(features[indices] @ w + b >= 0, 1, -1)
            score = np.mean(pred == labels)
            water_score = np.mean(pred[labels == 1] == 1)
            land_score = np.mean(pred[labels == -1] == -1)
            records.append([window, len(y), C, score, water_score, land_score])
            print(f"window={window}, samples={len(y)}, C={C}: validation={score:.4%}", flush=True)
            if (window, C) == (21, 1):
                before = ((features @ w + b >= 0).reshape(image.height, image.width) * 255).astype(np.uint8)
            if score > best_score:
                best_score = score
                best = records[-1]
                after = ((features @ w + b >= 0).reshape(image.height, image.width) * 255).astype(np.uint8)
    with open(output / "tuning.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["window", "samples", "C", "validation_accuracy", "water_recall", "land_recall"])
        writer.writerows(records)
    Image.fromarray(before).save(output / "before_tuning_mask.png")
    Image.fromarray(after).save(output / "after_tuning_mask.png")
    save_compare(image, [before, after], output / "tuning_compare.png", ("Input", "Default parameters", "Best parameters"))
    print("Best [window, samples, C, accuracy, water recall, land recall]:", best)


if __name__ == "__main__":
    main()
