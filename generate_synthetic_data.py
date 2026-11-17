"""
Synthetic Hand Landmark Dataset Generator for Sign Language Detection
=====================================================================
Generates canonical 3D MediaPipe landmark vectors (21 points x [x, y, z] = 63 floats)
for all 28 sign classes (A-Z, space, fullstop) with configurable perturbation.
"""

import os
import argparse
import numpy as np

LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "space", "fullstop"
]

WRIST = np.array([0.50, 0.80, 0.0])
MCP_BASES = {
    "thumb":  [0.42, 0.72, -0.02],
    "index":  [0.44, 0.55, -0.01],
    "middle": [0.50, 0.53, 0.00],
    "ring":   [0.56, 0.55, 0.01],
    "pinky":  [0.61, 0.58, 0.02]
}


def generate_finger(mcp, direction, length=0.08):
    p0 = np.array(mcp)
    d = np.array(direction) / (np.linalg.norm(direction) + 1e-8)
    p1 = p0 + d * (length * 0.4)
    p2 = p1 + d * (length * 0.35)
    p3 = p2 + d * (length * 0.3)
    return [p0, p1, p2, p3]


def generate_thumb(t_type):
    cmc = np.array([0.45, 0.75, -0.01])
    if t_type == "up":
        return [cmc, np.array([0.42, 0.68, -0.02]), np.array([0.40, 0.60, -0.02]), np.array([0.38, 0.52, -0.02])]
    elif t_type == "out":
        return [cmc, np.array([0.40, 0.73, -0.02]), np.array([0.34, 0.72, -0.02]), np.array([0.28, 0.72, -0.02])]
    elif t_type == "across":
        return [cmc, np.array([0.46, 0.70, 0.02]), np.array([0.50, 0.65, 0.03]), np.array([0.54, 0.62, 0.03])]
    elif t_type == "tucked_side":
        return [cmc, np.array([0.42, 0.70, -0.01]), np.array([0.43, 0.64, 0.01]), np.array([0.44, 0.58, 0.02])]
    elif t_type == "down":
        return [cmc, np.array([0.44, 0.78, 0.01]), np.array([0.43, 0.82, 0.02]), np.array([0.42, 0.86, 0.02])]
    elif t_type == "curled":
        return [cmc, np.array([0.45, 0.70, 0.01]), np.array([0.47, 0.66, 0.02]), np.array([0.48, 0.64, 0.02])]
    elif t_type == "spread":
        return [cmc, np.array([0.38, 0.72, -0.02]), np.array([0.32, 0.66, -0.03]), np.array([0.26, 0.60, -0.03])]
    else:
        return [cmc, np.array([0.42, 0.70, 0.0]), np.array([0.44, 0.65, 0.01]), np.array([0.46, 0.62, 0.02])]


def get_canonical(gesture):
    up = [0.0, -1.0, 0.0]
    fist = [0.0, 0.6, 0.05]
    curve = [0.0, -0.5, 0.4]
    spread_idx = [-0.3, -1.0, 0.0]
    spread_pky = [0.4, -1.0, 0.0]
    side = [-1.0, 0.0, 0.0]

    t_type = "tucked_side"
    f_dirs = [fist, fist, fist, fist]

    if gesture == "A":
        t_type, f_dirs = "up", [fist, fist, fist, fist]
    elif gesture == "B":
        t_type, f_dirs = "across", [up, up, up, up]
    elif gesture == "C":
        t_type, f_dirs = "spread", [curve, curve, curve, curve]
    elif gesture == "D":
        t_type, f_dirs = "across", [up, fist, fist, fist]
    elif gesture == "E":
        t_type, f_dirs = "down", [fist, fist, fist, fist]
    elif gesture == "F":
        t_type, f_dirs = "across", [curve, up, up, up]
    elif gesture == "G":
        t_type, f_dirs = "side", [side, fist, fist, fist]
    elif gesture == "H":
        t_type, f_dirs = "tucked_side", [side, side, fist, fist]
    elif gesture == "I":
        t_type, f_dirs = "across", [fist, fist, fist, up]
    elif gesture == "J":
        t_type, f_dirs = "across", [fist, fist, fist, curve]
    elif gesture == "K":
        t_type, f_dirs = "up", [up, [0.2, -1.0, 0.0], fist, fist]
    elif gesture == "L":
        t_type, f_dirs = "out", [up, fist, fist, fist]
    elif gesture == "M":
        t_type, f_dirs = "across", [[0.0, 0.7, 0.05], [0.0, 0.7, 0.05], [0.0, 0.7, 0.05], fist]
    elif gesture == "N":
        t_type, f_dirs = "across", [[0.0, 0.7, 0.05], [0.0, 0.7, 0.05], fist, fist]
    elif gesture == "O":
        t_type, f_dirs = "across", [curve, curve, curve, curve]
    elif gesture == "P":
        t_type, f_dirs = "down", [[0.0, 1.0, 0.0], [0.3, 0.9, 0.0], fist, fist]
    elif gesture == "Q":
        t_type, f_dirs = "down", [[0.0, 1.0, 0.0], fist, fist, fist]
    elif gesture == "R":
        t_type, f_dirs = "across", [[0.05, -1.0, 0.0], [-0.05, -1.0, 0.0], fist, fist]
    elif gesture == "S":
        t_type, f_dirs = "across", [fist, fist, fist, fist]
    elif gesture == "T":
        t_type, f_dirs = "up", [fist, fist, fist, fist]
    elif gesture == "U":
        t_type, f_dirs = "across", [up, up, fist, fist]
    elif gesture == "V":
        t_type, f_dirs = "across", [[-0.2, -1.0, 0.0], [0.2, -1.0, 0.0], fist, fist]
    elif gesture == "W":
        t_type, f_dirs = "across", [[-0.3, -1.0, 0.0], [0.0, -1.0, 0.0], [0.3, -1.0, 0.0], fist]
    elif gesture == "X":
        t_type, f_dirs = "across", [curve, fist, fist, fist]
    elif gesture == "Y":
        t_type, f_dirs = "spread", [fist, fist, fist, spread_pky]
    elif gesture == "Z":
        t_type, f_dirs = "across", [side, fist, fist, fist]
    elif gesture == "space":
        t_type, f_dirs = "spread", [spread_idx, up, up, spread_pky]
    elif gesture == "fullstop":
        t_type, f_dirs = "fist", [fist, fist, fist, fist]

    pts = [WRIST]
    pts.extend(generate_thumb(t_type))
    pts.extend(generate_finger(MCP_BASES["index"], f_dirs[0]))
    pts.extend(generate_finger(MCP_BASES["middle"], f_dirs[1]))
    pts.extend(generate_finger(MCP_BASES["ring"], f_dirs[2]))
    pts.extend(generate_finger(MCP_BASES["pinky"], f_dirs[3]))

    return np.array(pts).flatten()


def generate_dataset(output_dir="sample_dataset", samples_per_class=50, noise_std=0.008, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    total = 0

    for lbl in LABELS:
        class_dir = os.path.join(output_dir, lbl)
        os.makedirs(class_dir, exist_ok=True)
        canonical = get_canonical(lbl)

        for i in range(samples_per_class):
            noise = rng.normal(0, noise_std, size=canonical.shape)
            sample = np.clip(canonical + noise, -1.0, 2.0)
            np.save(os.path.join(class_dir, f"sample_{i:03d}.npy"), sample)
            total += 1

    print(f"Generated {total} synthetic landmark vectors across {len(LABELS)} classes in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ASL hand landmarks")
    parser.add_argument("--output", default="sample_dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=50, help="Samples per class")
    parser.add_argument("--noise", type=float, default=0.008, help="Gaussian noise standard deviation")
    args = parser.parse_args()
    generate_dataset(output_dir=args.output, samples_per_class=args.samples, noise_std=args.noise)
