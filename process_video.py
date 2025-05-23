import numpy as np
import cv2
import os
from glob import glob
import time

# -------- Parameters --------
NUM_VIEWS = 60
RADIUS = 3.0
CAMERA_HEIGHT = 1.5
BASE_OVERLAY_SIZE = 64
FPS = 30
BUFFER_DURATION = 1.0

BG_PATH = "grass.jpg"  # Background image path
BG_Y = 0

OUTPUT_PATH = "camera_orbit_multiobject.mp4"
HLEN = 100

# Each object: (folder_name, 3D_position)
OBJECTS_INFO = [
    ("object-videos/objectA_views", np.array([0.0, 0.0, 0.0])),   # Object A (center)
    ("object-videos/objectB_views", np.array([1.0, 0.0, 0.0])),   # Object B
]

# -------- Load images --------
def load_images_from_folder(folder, max_images=None):
    paths = sorted(glob(os.path.join(folder, '*.png')))
    if max_images:
        paths = paths[:max_images]
    images = [cv2.imread(p) for p in paths]
    return images

# -------- Camera orbit poses --------
def generate_orbit_camera_poses(radius=2.0, num_views=60, height=0.5):
    poses = []
    for theta in np.linspace(0, 2 * np.pi, num_views, endpoint=False):
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        y = height
        camera_position = np.array([x, y, z])
        target = np.array([0, 0, 0])
        forward = (target - camera_position)
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=1)
        T = -R.T @ camera_position
        extrinsic = np.concatenate([R.T, T.reshape(3, 1)], axis=1)
        poses.append(extrinsic)
    return poses

# -------- Project 3D point --------
def project_point(K, extrinsic, point3D):
    point_cam = extrinsic[:, :3] @ point3D + extrinsic[:, 3]
    if point_cam[2] <= 0:
        return None
    point_proj = K @ point_cam
    return (point_proj[:2] / point_proj[2]).astype(int)

# -------- Scale image based on distance --------
def get_scaled_overlay(image, distance, base_distance, base_size):
    scale = base_distance / distance
    size = int(base_size * scale)
    size = max(8, min(size, 512))
    return cv2.resize(image, (size, size)), size

# -------- Main --------
def main():
    # Load all objects
    objects = []
    for folder, position in OBJECTS_INFO:
        images = load_images_from_folder(folder, NUM_VIEWS)
        if len(images) != NUM_VIEWS:
            raise ValueError(f"Folder {folder} must contain exactly {NUM_VIEWS} images.")
        objects.append({
            "folder": folder,
            "position": position,
            "images": images
        })

    # Get resolution from first image
    IMG_SIZE = (objects[0]['images'][0].shape[1], objects[0]['images'][0].shape[0])  # (width, height)

    if not os.path.exists(BG_PATH):
        raise FileNotFoundError(f"Background image not found: {BG_PATH}")
    else:
        background = cv2.imread(BG_PATH)
        background = cv2.resize(background, (IMG_SIZE[0], IMG_SIZE[1]))
        if background is None:
            raise ValueError(f"Failed to load background image: {BG_PATH}")

    # Camera intrinsics
    focal_length = 0.7 * IMG_SIZE[0]
    K = np.array([
        [focal_length, 0, IMG_SIZE[0] / 2],
        [0, focal_length, IMG_SIZE[1] / 2],
        [0, 0, 1]
    ])

    orbit_poses = generate_orbit_camera_poses(RADIUS, NUM_VIEWS, CAMERA_HEIGHT)
    frames = []

    # For background warping
    pts_bg_w = np.array([
        [ HLEN, BG_Y,  HLEN],
        [ HLEN, BG_Y, -HLEN],
        [-HLEN, BG_Y, -HLEN],
        [-HLEN, BG_Y,  HLEN],
    ], dtype=np.float32)
    h, w = IMG_SIZE[1], IMG_SIZE[0]

    for i in range(NUM_VIEWS):
        pose = orbit_poses[i]
        frame = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        camera_position = -pose[:, :3].T @ pose[:, 3]
        
        render_list = []

        ### Background warping ###

        pts_bg_warped = []

        # src: [TL, BL, TR, BR]
        pts_bg_src = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1],
        ], dtype=int)

        dst = []
        for pt in pts_bg_w:
            pc = pose[:, :3] @ pt + pose[:, 3]
            p  = K @ pc
            uv = (p[:2] / p[2]).astype(int)
            uv = np.clip(uv, [0,0], [w-1, h-1])
            dst.append(uv)
        dst = np.stack(dst, axis=0)  # shape (4,2)

        tops = dst[np.argsort(dst[:, 1])][:2]
        bottoms = dst[np.argsort(dst[:, 1])][2:]

        tops = tops[np.argsort(tops[:, 0])]
        bottoms = bottoms[np.argsort(bottoms[:, 0])[::-1]]

        pts_bg_warped = np.concatenate([tops, bottoms])

        M_warp = cv2.getPerspectiveTransform(pts_bg_src.astype(np.float32), pts_bg_warped.astype(np.float32))
        bg_warped = cv2.warpPerspective(background, M_warp, (w, h))
        
        frame = bg_warped.copy()

        cv2.imshow("Background", bg_warped)
        cv2.waitKey(1)

        ### Backward warping ###

        for obj in objects:
            obj_pos = obj["position"]
            obj_image = obj["images"][i]
            dist = np.linalg.norm(obj_pos - camera_position)
            point2D = project_point(K, pose, obj_pos)

            if point2D is not None:
                REFERENCE_DISTANCE = 1.0  # constant, like a camera 1m away
                overlay_img, overlay_size = get_scaled_overlay(obj_image, dist, REFERENCE_DISTANCE, BASE_OVERLAY_SIZE)
                x, y = point2D
                x -= overlay_size // 2
                y -= overlay_size // 2

                if 0 <= x < IMG_SIZE[0] - overlay_size and 0 <= y < IMG_SIZE[1] - overlay_size:
                    render_list.append((dist, overlay_img, overlay_size, x, y))

        # Sort by distance (far → near)
        render_list.sort(reverse=True, key=lambda item: item[0])

        # Composite in correct order
        for _, overlay_img, overlay_size, x, y in render_list:
            roi = frame[y:y + overlay_size, x:x + overlay_size]
            frame[y:y + overlay_size, x:x + overlay_size] = cv2.addWeighted(
                roi, 0.3, overlay_img, 0.7, 0
            )

        frames.append(frame)

    # -------- Add start/end buffer --------
    buffer_frame_count = int(FPS * BUFFER_DURATION)
    frames = (
        [frames[0]] * buffer_frame_count +
        frames +
        [frames[-1]] * buffer_frame_count
    )

    # -------- Write video --------
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, IMG_SIZE)
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"✅ Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()