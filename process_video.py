import numpy as np
import cv2
import os
import sys
import shutil
from glob import glob
import torch
from tqdm import tqdm

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

# -------- Parameters --------
NUM_VIEWS = 124                # 추출할 프레임 수
FRAME_RESIZE = (1024, 1024)    # None 이면 리사이즈 안 함

UPSCALE_FACTOR = 3             # 거리 감쇠용 인자
RADIUS = 13.0                  # 카메라 궤도 반경
CAMERA_HEIGHT = 5              # 카메라 높이 (world y)
FPS = 30                       # 출력 비디오 fps
BUFFER_DURATION = 1.0          # 시작/끝 버퍼 지속 시간(초)

BG_PATH = "grass.jpg"          # 바닥 텍스처 이미지 경로
BG_Y = 0                       # 바닥 평면의 y 좌표
BG_LEN = 6.0

ENABLE_BG_CUT = True
OUTPUT_PATH = "camera_orbit_multiobject_upscaled.mp4"


def extract_frames(video_path, out_folder, max_frames):
    if ENABLE_BG_CUT:
        seg_net = TracerUniversalB7(device='cpu', batch_size=1)
        fba = FBAMatting(device='cpu', input_tensor_size=2048, batch_size=1)
        trimap = TrimapGenerator()
        preprocessing = PreprocessingStub()
        postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cpu')
        interface = Interface(pre_pipe=preprocessing, post_pipe=None, seg_pipe=seg_net)

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    print(f"Extracting frames from {video_path} to {out_folder}...")

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] {video_path} frame {i} unreadable.")
            break
        if FRAME_RESIZE is not None:
            frame = cv2.resize(frame, FRAME_RESIZE, interpolation=cv2.INTER_CUBIC)
        frame_path = os.path.join(out_folder, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    cap.release()

    if not ENABLE_BG_CUT:
        frames = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in frame_paths]
        if any(f is None for f in frames):
            raise RuntimeError("Failed to load some frames.")
        return frames

    # normal map 경로 유추
    vpath, vname = os.path.split(video_path)
    normal_path = os.path.join(vpath, vname[:-4].replace('test', 'test-normal'))
    if not os.path.exists(normal_path):
        raise RuntimeError(f"Normal map directory {normal_path} does not exist.")
    normal_paths = sorted(glob(os.path.join(normal_path, "*.png")))
    normal_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    if len(normal_paths) != len(frame_paths):
        raise RuntimeError(f"Not enough normal maps found in {normal_path}.")
    print(f"Found {len(normal_paths)} normal maps in {normal_path}.")

    cleaned = []
    for i, frame_path in enumerate(frame_paths):
        normal_img = cv2.imread(normal_paths[i], cv2.IMREAD_UNCHANGED)
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if img is None or normal_img is None:
            raise RuntimeError(f"Frame {i} or normal map {i} load failed.")
        # alpha 채널 추가
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))
        # 검은 픽셀 투명화
        black_mask = np.all(normal_img[:, :, :3] < 10, axis=2)
        img[black_mask, 3] = 0
        cv2.imwrite(frame_path, img)
        cleaned.append(img)
    return cleaned


def load_images_from_dir(dir_path, max_frames):
    parent_dir, img_path = os.path.split(dir_path)
    normal_path = os.path.join(parent_dir, img_path.replace('test', 'test-normal'))
    if not os.path.exists(normal_path):
        raise RuntimeError(f"Normal map directory {normal_path} does not exist.")
    normal_paths = sorted(glob(os.path.join(normal_path, "*.png")))
    normal_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    paths = sorted(glob(os.path.join(dir_path, "*.png")))
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
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
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        C = np.array([x, height, z])
        forward = (np.array([0.,0.,0.]) - C)
        forward /= np.linalg.norm(forward)
        up_vec = np.array([0.,1.,0.])
        right = np.cross(up_vec, forward); right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R_cam = np.stack([right, up, forward], axis=1).T
        t_cam = -R_cam @ C
        poses.append((np.concatenate([R_cam, t_cam.reshape(3,1)], axis=1), C))
    return poses


def project_point(K, extrinsic, point3D):
    pc = extrinsic[:,:3] @ point3D + extrinsic[:,3]
    proj = K @ pc
    return (proj[:2] / proj[2]).astype(np.int32)


def get_scaled_overlay(img, dist, base_dist, base_size):
    sz = int(base_size * (base_dist / dist) * UPSCALE_FACTOR)
    sz = max(8, min(sz, 512 * UPSCALE_FACTOR))
    return cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC), sz


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

    frames1, frames2, frames3 = frames_list

    # --- 여기서 객체별 base_overlay_size 지정 ---
    objects = [
        {'position': np.array([0.0, BG_Y, 0.0]), 'images': frames1, 'base_overlay_size': 1024},
        {'position': np.array([1.0, BG_Y, 2.0]), 'images': frames2, 'base_overlay_size': 512},
        {'position': np.array([-5.0, BG_Y, 5.0]), 'images': frames3, 'base_overlay_size': 512},
    ]

    if FRAME_RESIZE is not None:
        frames1 = [cv2.resize(f, FRAME_RESIZE, interpolation=cv2.INTER_CUBIC) for f in frames1]
        frames2 = [cv2.resize(f, FRAME_RESIZE, interpolation=cv2.INTER_CUBIC) for f in frames2]
        frames3 = [cv2.resize(f, FRAME_RESIZE, interpolation=cv2.INTER_CUBIC) for f in frames3]

    h0, w0 = frames1[0].shape[:2]
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
            # 객체별로 지정한 base_overlay_size 사용
            overlay, sz = get_scaled_overlay(
                obj['images'][idx],
                dist,
                base_dist=1.0,
                base_size=obj['base_overlay_size']
            )
            x, y = p2d - sz // 2
            if 0 <= x <= w0 - sz and 0 <= y <= h0 - sz:
                render_list.append((dist, overlay, sz, x, y))

        # alpha blending
        for _, ov, sz, x, y in sorted(render_list, key=lambda x: x[0], reverse=True):
            overlay_rgb = cv2.flip(ov[:, :, :3].astype(float), -1)
            try:
                alpha_mask = ov[:, :, 3].astype(float) / 255.0
            except IndexError:
                alpha_mask = np.ones((sz, sz), dtype=float)
            alpha = cv2.flip(cv2.merge([alpha_mask]*3), -1)

            roi = frame[y:y+sz, x:x+sz].astype(float)
            blended = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)
            frame[y:y+sz, x:x+sz] = blended

        out_frames.append(frame)

    for i in range(len(out_frames)):
        out_frames[i] = cv2.flip(out_frames[i], -1)  # Flip vertically

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
