import numpy as np
import cv2
import os
import sys
import shutil
from glob import glob
import torch
from tqdm import tqdm

# -------- Parameters --------
NUM_VIEWS = 60                # 추출할 프레임 수
FRAME_RESIZE = (1024, 1024)    # None 이면 리사이즈 안 함
VIDEO_SIZE = (1024, 1024)

RADIUS = 6.0                  # 카메라 궤도 반경
CAMERA_HEIGHT = 2              # 카메라 높이 (world y)
FPS = 30                       # 출력 비디오 fps
BUFFER_DURATION = 1.0          # 시작/끝 버퍼 지속 시간(초)

BG_PATH = "grass.jpg"          # 바닥 텍스처 이미지 경로
BG_Y = 0                       # 바닥 평면의 y 좌표
BG_LEN = 4.6

ENABLE_BG_CUT = True
OUTPUT_PATH = "camera_orbit_multiobject_upscaled.mp4"


def get_refined_list(lst):
    L = [lst[10]] + lst[16:30] + [lst[30]] + lst[47:61] + [lst[70]] + lst[78:92] + [lst[92]] + lst[109:123]
    L.reverse()
    return L

def load_images_from_dir(dir_path, max_frames):
    parent_dir, img_path = os.path.split(dir_path)
    normal_path = os.path.join(parent_dir, img_path.replace('test', 'test-normal'))
    if not os.path.exists(normal_path):
        raise RuntimeError(f"Normal map directory {normal_path} does not exist.")
    normal_paths = sorted(glob(os.path.join(normal_path, "*.png")))
    normal_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    normal_paths = get_refined_list(normal_paths)

    paths = sorted(glob(os.path.join(dir_path, "*.png")))
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    paths = get_refined_list(paths)
    if len(paths) < max_frames:
        print(f"[Warning] Directory {dir_path} has only {len(paths)} images (requested {max_frames}).")

    imgs = []
    for i, p in enumerate(paths[:max_frames]):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        normal_img = cv2.imread(normal_paths[i], cv2.IMREAD_UNCHANGED)
        if img is None or normal_img is None:
            print(f"[Warning] Failed to load image {p}.")
            continue
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))
        black_mask = np.all(normal_img[:, :, :3] < 10, axis=2)
        img[black_mask, 3] = 0
        imgs.append(img)
    return imgs


def generate_orbit_camera_poses(radius, num_views, height):
    poses = []
    for theta in np.linspace(0, 2*np.pi, num_views, endpoint=False):
        x = radius * np.cos(-theta)
        z = radius * np.sin(-theta)
        C = np.array([x, height, z])
        forward = (np.array([0.,0.,0.]) - C)
        forward /= np.linalg.norm(forward)
        down_vec = np.array([0.,-1.,0.])
        right = np.cross(down_vec, forward); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R_cam = np.stack([right, down, forward], axis=1).T
        t_cam = -R_cam @ C
        poses.append((np.concatenate([R_cam, t_cam.reshape(3,1)], axis=1), C))
    return poses


def project_point(K, extrinsic, point3D):
    pc = extrinsic[:,:3] @ point3D + extrinsic[:,3]
    proj = K @ pc
    return (proj[:2] / proj[2]).astype(np.int32)


def get_scaled_overlay(image, distance, base_distance, base_size):
    scale = base_distance / distance
    size = int(base_size * scale)
    size = max(8, min(size, 512))
    return cv2.resize(image, (size, size)), size


def compute_ground_homography(K, extrinsic, bg_size, w, h):
    pts3d = np.array([
        [-bg_size, BG_Y, -bg_size], [ bg_size, BG_Y, -bg_size],
        [ bg_size, BG_Y,  bg_size], [-bg_size, BG_Y,  bg_size]
    ])
    pts2d = [project_point(K, extrinsic, pt) for pt in pts3d]
    pts2d.sort(key=lambda x: (x[1], x[0]))
    pts = np.array(pts2d, dtype=np.float32)
    tl, tr, bl, br = pts
    dst = np.array([[0,0],[w,0],[0,h],[w,h]], dtype=np.float32)
    src = np.array([tl, tr, bl, br], dtype=np.float32)
    try:
        return cv2.getPerspectiveTransform(dst, src)
    except:
        if len(pts) >= 3:
            M = cv2.getAffineTransform(dst[:3], src[:3])
            return np.vstack([M, [0,0,1]])
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_or_dir1> <video_or_dir2>")
        sys.exit(1)
    inputs = sys.argv[1:]
    frames_list = []
    for i, inp in enumerate(inputs):
        if os.path.isdir(inp):
            frames = load_images_from_dir(inp, NUM_VIEWS)
        else:
            folder = f"frames_{i}"
            frames = extract_frames(inp, folder, NUM_VIEWS)
        if len(frames) < NUM_VIEWS:
            raise RuntimeError(f"Not enough frames loaded for {inp}.")
        frames_list.append(frames)

    frames1, frames2, frames3, frames4 = frames_list

    # --- 여기서 객체별 base_overlay_size 지정 ---
    objects = [
        # Scarecrow
        {'position': np.array([1.0, BG_Y, 1.0]), 'images': frames1, 'base_overlay_size': 1600, 'frame_size': (1200, 1200)},
        # Boy
        {'position': np.array([1.0, BG_Y, 2.3]), 'images': frames2, 'base_overlay_size': 850, 'frame_size': (600, 600)},
        # Castle
        {'position': np.array([-1.0, BG_Y, -1.0]), 'images': frames3, 'base_overlay_size': 4000, 'frame_size': (3000, 3500)},
        # Dog
        {'position': np.array([2.0, BG_Y, 1.5]), 'images': frames4, 'base_overlay_size': 600, 'frame_size': (600, 600)},
    ]

    h0, w0 = VIDEO_SIZE
    bg = cv2.imread(BG_PATH, cv2.IMREAD_COLOR)
    bg = cv2.resize(bg, (w0, h0), interpolation=cv2.INTER_CUBIC)

    focal = 0.7 * w0
    K = np.array([[focal, 0, w0/2],
                  [0, focal, h0/2],
                  [0,     0,    1 ]])

    poses = generate_orbit_camera_poses(RADIUS, NUM_VIEWS, CAMERA_HEIGHT)
    out_frames = []

    for idx, (extrinsic, C) in enumerate(poses):
        H = compute_ground_homography(K, extrinsic, BG_LEN, w0, h0)
        if H is None:
            raise RuntimeError("Failed to compute homography.")
        bg_warped = cv2.warpPerspective(bg, H, (w0, h0))
        frame = bg_warped.copy()

        render_list = []
        for obj in objects:
            dist = np.linalg.norm(C - obj['position'])
            p2d = project_point(K, extrinsic, obj['position'])
            
            # 나머지 코드는 동일하게 유지
            overlay, sz = get_scaled_overlay(
                obj['images'][idx],
                dist,
                1.0,
                base_size=obj['base_overlay_size']  # base_overlay_size
            )
            x, y = p2d - sz // 2
            if 0 <= x <= w0 - sz and 0 <= y <= h0 - sz:
                render_list.append((dist, overlay, sz, x, y))

        # alpha blending
        for _, ov, sz, x, y in sorted(render_list, key=lambda x: x[0], reverse=True):
            overlay_rgb = ov[:, :, :3].astype(float)
            try:
                alpha_mask = ov[:, :, 3].astype(float) / 255.0
            except IndexError:
                alpha_mask = np.ones((sz, sz), dtype=float)
            alpha = cv2.merge([alpha_mask]*3)

            roi = frame[y:y+sz, x:x+sz].astype(float)
            blended = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)
            frame[y:y+sz, x:x+sz] = blended

        out_frames.append(frame)

    buf = int(FPS * BUFFER_DURATION)
    frames_buf = [out_frames[0]] * buf + out_frames + [out_frames[-1]] * buf
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (w0, h0)
    )
    for f in frames_buf:
        writer.write(f)
    writer.release()
    print(f"✅ Saved merged video: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
