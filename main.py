import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from svm_primal_image_seg import solve_primal
from svm_dual_image_seg import solve_dual
from svm_image_utils import select_samples, prepare_data, save_samples, save_compare


def main():
    folder = Path(__file__).resolve().parent.parent
    output = folder / "output"
    output.mkdir(exist_ok=True)
    image = Image.open(folder / "00000341.jpg").convert("RGB")
    C = 1.0
    window = 21
    samples = select_samples(image)
    # if samples is None:
    #     return
    features, X, y = prepare_data(image, samples, window)
    # save_samples(image, samples, output / "samples.png")
    np.savetxt(output / "samples.csv", samples, fmt="%d", delimiter=",", header="x,y,label", comments="")

    start = perf_counter()
    w1, b1 = solve_primal(X, y, C)
    time1 = perf_counter() - start
    start = perf_counter()
    w2, b2, a = solve_dual(X, y, C)
    time2 = perf_counter() - start

    masks = []
    results = {"image_size": list(image.size), "samples": len(y), "features": X.shape[1], "C": C, "window": window}
    for name, w, b, elapsed in [("primal", w1, b1, time1), ("dual", w2, b2, time2)]:
        start = perf_counter()
        mask = (features @ w + b >= 0).reshape(image.height, image.width)
        predict_time = perf_counter() - start
        mask = mask.astype(np.uint8) * 255
        Image.fromarray(mask).save(output / (name + "_mask.png"))
        masks.append(mask)
        loss = np.maximum(0, 1 - y * (X @ w + b))
        accuracy = np.mean(np.where(X @ w + b >= 0, 1, -1) == y)
        results[name] = {
            "train_seconds": elapsed, "predict_seconds": predict_time,
            "train_accuracy": float(accuracy), "primal_objective": float(0.5 * w @ w + C * loss.sum()),
            "w": w.tolist(), "b": b
        }
        print(f"{name}: 训练 {elapsed:.4f} 秒，预测 {predict_time:.4f} 秒，训练准确率 {accuracy:.2%}")

    # dual_value = float(a.sum() - 0.5 * w2 @ w2)
    # results["dual_objective"] = dual_value
    # results["primal_dual_gap"] = results["primal"]["primal_objective"] - dual_value
    # results["dual_solution_gap"] = results["dual"]["primal_objective"] - dual_value
    # results["dual_equality_error"] = float(abs(a @ y))
    # results["support_vectors"] = int(np.sum(a > 1e-6))
    # results["max_w_difference"] = float(np.max(np.abs(w1 - w2)))
    # results["b_difference"] = float(abs(b1 - b2))
    # results["different_pixels"] = int(np.sum(masks[0] != masks[1]))
    # results["mask_agreement"] = float(np.mean(masks[0] == masks[1]))
    # with open(output / "metrics.json", "w", encoding="utf-8") as file:
    #     json.dump(results, file, ensure_ascii=False, indent=2)
    save_compare(image, masks, output / "compare.png")
    # print(f"分割一致率：{results['mask_agreement']:.4%}，不同像素：{results['different_pixels']}")
    # print(f"原始-对偶目标值差：{results['primal_dual_gap']:.3e}")
    # print("没有完整人工标注，训练准确率和两种方法的一致率都不是整图分割准确率。")
    # print("结果目录：", output)


if __name__ == "__main__":
    main()
