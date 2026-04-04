import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import time

# --- Path to your 3D coords CSV ---
base_path = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(base_path, '..', 'output_data', 'hands_3d_coords.csv')

# --- MediaPipe hand skeleton connections (21 landmarks) ---
HAND_CONNECTIONS = [
    # Wrist to finger bases
    (0, 1), (0, 5), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

HAND_COLORS = ['#00CFFF', '#FF6B6B']   # cyan = hand 0, red = hand 1
CONN_COLORS = ['#0088AA', '#AA3333']

# --- Playback speed ---
FRAME_INTERVAL_MS = 50   # milliseconds between frames (~20 fps)


# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_3d_csv(filepath):
    frames = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            vals = [float(x) for x in row]
            frame_idx = int(vals[0])
            timestamp  = int(vals[1])
            # hands: shape (2, 21, 3)
            hands = np.array(vals[2:]).reshape(2, 21, 3)
            frames.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'hands': hands,
            })
    return frames


# ── Compute axis limits from all data ─────────────────────────────────────────
def compute_limits(frames):
    all_pts = []
    for f in frames:
        pts = f['hands'].reshape(-1, 3)
        # ignore zero points (no detection)
        valid = pts[~np.all(pts == 0, axis=1)]
        if len(valid):
            all_pts.append(valid)
    if not all_pts:
        return (-1, 1), (-1, 1), (-1, 1)
    all_pts = np.vstack(all_pts)
    pad = 50  # mm padding
    return (
        (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad),
        (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad),
        (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad),
    )


# ── Draw one frame ─────────────────────────────────────────────────────────────
def draw_frame(ax, frame_data, xlim, ylim, zlim):
    ax.cla()

    ax.set_facecolor('#0D0D1A')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel('X (mm)', color='#888888', fontsize=8)
    ax.set_ylabel('Y (mm)', color='#888888', fontsize=8)
    ax.set_zlabel('Z (mm)', color='#888888', fontsize=8)
    ax.tick_params(colors='#555555', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1A1A2E')
    ax.yaxis.pane.set_edgecolor('#1A1A2E')
    ax.zaxis.pane.set_edgecolor('#1A1A2E')
    ax.grid(True, color='#1A1A2E', linewidth=0.5)

    for h in range(2):
        pts = frame_data['hands'][h]  # (21, 3)

        # Skip hand if all zeros (no detection)
        if np.all(pts == 0):
            continue

        pt_color   = HAND_COLORS[h]
        conn_color = CONN_COLORS[h]
        label      = f'Hand {h}'

        # Draw connections
        for (i, j) in HAND_CONNECTIONS:
            p1, p2 = pts[i], pts[j]
            if np.all(p1 == 0) or np.all(p2 == 0):
                continue
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=conn_color, linewidth=1.2, alpha=0.7
            )

        # Draw landmark points
        valid_mask = ~np.all(pts == 0, axis=1)
        visible = pts[valid_mask]
        ax.scatter(
            visible[:, 0], visible[:, 1], visible[:, 2],
            c=pt_color, s=18, depthshade=True,
            label=label, zorder=5
        )

        # Highlight wrist (landmark 0) larger
        wrist = pts[0]
        if not np.all(wrist == 0):
            ax.scatter(*wrist, c=pt_color, s=60, marker='*', zorder=6)

    ax.set_title(
        f"Frame {frame_data['frame_idx']}  |  t = {frame_data['timestamp']} ms",
        color='#CCCCCC', fontsize=10, pad=8
    )
    ax.legend(loc='upper left', fontsize=8, facecolor='#111111',
              labelcolor='white', framealpha=0.5)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Run triangulate_3d.py first to generate hands_3d_coords.csv"
        )

    frames = load_3d_csv(CSV_PATH)
    print(f"Loaded {len(frames)} frames.")

    xlim, ylim, zlim = compute_limits(frames)

    fig = plt.figure(figsize=(10, 7), facecolor='#0D0D1A')
    ax  = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.05)

    # Controls state
    state = {'idx': 0, 'playing': True}

    def update(_):
        if state['playing']:
            draw_frame(ax, frames[state['idx']], xlim, ylim, zlim)
            state['idx'] = (state['idx'] + 1) % len(frames)

    def on_key(event):
        if event.key == ' ':                          # space = pause/play
            state['playing'] = not state['playing']
        elif event.key == 'right' and not state['playing']:
            state['idx'] = min(state['idx'] + 1, len(frames) - 1)
            draw_frame(ax, frames[state['idx']], xlim, ylim, zlim)
            fig.canvas.draw()
        elif event.key == 'left' and not state['playing']:
            state['idx'] = max(state['idx'] - 1, 0)
            draw_frame(ax, frames[state['idx']], xlim, ylim, zlim)
            fig.canvas.draw()
        elif event.key == 'r':                        # r = restart
            state['idx'] = 0

    fig.canvas.mpl_connect('key_press_event', on_key)

    ani = animation.FuncAnimation(
        fig, update,
        interval=FRAME_INTERVAL_MS,
        cache_frame_data=False
    )

    print("\nControls:")
    print("  SPACE      — pause / resume")
    print("  LEFT/RIGHT — step frame (while paused)")
    print("  R          — restart from frame 0")
    print("  Drag       — rotate 3D view\n")

    plt.show()


if __name__ == "__main__":
    main()
