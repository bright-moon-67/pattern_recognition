from pathlib import Path

import numpy as np
from PIL import Image

from svm_primal_image_seg import solve_primal
from svm_dual_image_seg import solve_dual
from svm_rbf_image_seg import solve_rbf, predict_rbf
from svm_image_utils import select_samples, prepare_data, save_compare


def main():
    folder = Path(__file__).resolve().parent.parent
    output = folder / "output"
    output.mkdir(exist_ok=True)
    image = Image.open(folder / "00000341.jpg").convert("RGB")
    C, gamma, window = 1.0, 0.1, 21
    samples = select_samples(image)
    features, X, y = prepare_data(image, samples, window)
    np.savetxt(output / "samples.csv", samples, fmt="%d", delimiter=",", header="x,y,label", comments="")

    w1, b1 = solve_primal(X, y, C)
    w2, b2, _ = solve_dual(X, y, C)
    a, b = solve_rbf(X, y, C, gamma)
    masks = [(features @ w1 + b1 >= 0).astype(np.uint8) * 255,
             (features @ w2 + b2 >= 0).astype(np.uint8) * 255,
             predict_rbf(features, X, y, a, b, gamma)]
    for i, name in enumerate(["primal", "dual", "rbf"]):
        masks[i] = masks[i].reshape(image.height, image.width)
        Image.fromarray(masks[i]).save(output / (name + "_mask.png"))
    save_compare(image, masks, output / "compare.png")
    print("四图对比：", output / "compare.png")


if __name__ == "__main__":
    main()
