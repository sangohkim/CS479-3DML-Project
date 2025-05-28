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
NUM_VIEWS = 60           # 추출할 프레임 수
UPSCALE_FACTOR = 3       # 해상도 업스케일 비율
RADIUS = 16.0            # 카메라 궤도 반경
CAMERA_HEIGHT = 5        # 카메라 높이 (world y)
BASE_OVERLAY_SIZE = 512  # 객체 오버레이 기본 사이즈
FPS = 30                 # 출력 비디오 fps
BUFFER_DURATION = 1.0    # 시작/끝 버퍼 지속 시간(초)
BG_PATH = "grass.jpg"    # 바닥 텍스처 이미지 경로
BG_Y = 0                 # 바닥 평면의 y 좌표
BG_LEN = 6.0
OUTPUT_PATH = "camera_orbit_multiobject_upscaled.mp4"


def extract_frames(video_path, out_folder, max_frames, scale_factor):
    seg_net = TracerUniversalB7(device='cpu', batch_size=1)
    fba = FBAMatting(device='cpu', input_tensor_size=2048, batch_size=1)
    trimap = TrimapGenerator()
    preprocessing = PreprocessingStub()
    postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cpu')
    interface = Interface(pre_pipe=preprocessing, post_pipe=None, seg_pipe=seg_net)

    # 기존 폴더 삭제
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    print(f"Extracting frames from {video_path} to {out_folder}...")
    os.makedirs(out_folder)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] {video_path} frame {i} unreadable.")
            break
        # 해상도 업스케일
        h, w = frame.shape[:2]
        frame = cv2.resize(frame,
                           (int(w*scale_factor), int(h*scale_factor)),
                           interpolation=cv2.INTER_CUBIC)
        frame_path = os.path.join(out_folder, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    cap.release()

    # 배경 제거 (PIL RGBA -> PNG with alpha)
    frames_wo_bg = interface(frame_paths)
    cleaned = []
    for i, frame_wo_bg in enumerate(frames_wo_bg):
        if frame_wo_bg is not None:
            frame_wo_bg.save(frame_paths[i])
            # RGBA 채널을 유지하며 불러오기
            img = cv2.imread(frame_paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Frame {i} load failed after segmentation.")
            cleaned.append(img)
        else:
            raise RuntimeError(f"Frame {i} segmentation failed.")
    return cleaned


def load_images_from_dir(dir_path, max_frames):
    imgs = []
    paths = sorted(glob(os.path.join(dir_path, "*.png")))
    if len(paths) < max_frames:
        print(f"[Warning] Directory {dir_path} has only {len(paths)} images (requested {max_frames}).")
    for p in paths[:max_frames]:
        # RGBA 채널을 유지하며 불러오기
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[Warning] Failed to load image {p}.")
        else:
            imgs.append(img)
    return imgs


def generate_orbit_camera_poses(radius, num_views, height):
    poses = []
    for theta in np.linspace(0, 2*np.pi, num_views, endpoint=False):
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        C = np.array([x, height, z])
        target = np.array([0.0, 0.0, 0.0])
        forward = (target - C); forward /= np.linalg.norm(forward)
        up_vec = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_vec, forward); right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R_cam = np.stack([right, up, forward], axis=1).T
        t_cam = -R_cam @ C
        extrinsic = np.concatenate([R_cam, t_cam.reshape(3,1)], axis=1)
        poses.append((extrinsic, C))
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
    ground_points = np.array([
        [-bg_size, BG_Y, -bg_size], [bg_size, BG_Y, -bg_size],
        [bg_size, BG_Y,  bg_size], [-bg_size, BG_Y,  bg_size]
    ])
    image_points = [project_point(K, extrinsic, pt) for pt in ground_points]
    # 정렬: y→x
    image_points = sorted(image_points, key=lambda x: (x[1], x[0]))
    pts = np.array(image_points, dtype=np.float32)
    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]
    dst = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
    src = np.array([tl, tr, bl, br], dtype=np.float32)
    try:
        return cv2.getPerspectiveTransform(dst, src)
    except:
        if len(pts) >= 3:
            M = cv2.getAffineTransform(dst[:3], src[:3])
            H = np.vstack([M, [0,0,1]])
            return H
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <video_or_dir1> <video_or_dir2>")
        sys.exit(1)
    inputs = sys.argv[1:]
    frames_list = []
    for i, inp in enumerate(inputs):
        if os.path.isdir(inp):
            frames = load_images_from_dir(inp, NUM_VIEWS)
        else:
            folder = f"frames_{i}"
            frames = extract_frames(inp, folder, NUM_VIEWS, UPSCALE_FACTOR)
        if len(frames) < NUM_VIEWS:
            raise RuntimeError(f"Not enough frames loaded for {inp}.")
        frames_list.append(frames)

    frames1, frames2 = frames_list
    objects = [
        {'position': np.array([0.0, BG_Y, 0.0]), 'images': frames1},
        {'position': np.array([2.0, BG_Y, 0.0]), 'images': frames2}
    ]

    h0, w0 = frames1[0].shape[:2]
    bg = cv2.imread(BG_PATH, cv2.IMREAD_COLOR)
    bg = cv2.resize(bg, (w0, h0), interpolation=cv2.INTER_CUBIC)

    focal = 0.7 * w0
    K = np.array([[focal, 0, w0/2],
                  [0, focal, h0/2],
                  [0,     0,     1]])
    poses = generate_orbit_camera_poses(RADIUS, NUM_VIEWS, CAMERA_HEIGHT)
    out_frames = []

    for extrinsic, C in poses:
        H = compute_ground_homography(K, extrinsic, BG_LEN, w0, h0)
        if H is None:
            raise RuntimeError("Failed to compute homography.")
        bg_warped = cv2.warpPerspective(bg, H, (w0, h0))
        frame = cv2.flip(bg_warped.copy(), -1)  # BGR

        render_list = []
        for obj in objects:
            dist = np.linalg.norm(C - obj['position'])
            p2d = project_point(K, extrinsic, obj['position'])
            overlay, sz = get_scaled_overlay(obj['images'][len(out_frames)], dist, 1.0, BASE_OVERLAY_SIZE)
            x, y = p2d - sz // 2
            if 0 <= x <= w0 - sz and 0 <= y <= h0 - sz:
                render_list.append((dist, overlay, sz, x, y))

        # alpha blending 적용
        for _, ov, sz, x, y in sorted(render_list, key=lambda x: x[0], reverse=True):
            # ov: BGRA (h, w, 4)
            overlay_rgb = ov[:, :, :3].astype(float)
            alpha_mask = ov[:, :, 3].astype(float) / 255.0  # (h, w)
            alpha = cv2.merge([alpha_mask, alpha_mask, alpha_mask])  # (h, w, 3)

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
