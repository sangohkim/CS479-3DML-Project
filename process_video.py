import numpy as np
import cv2
import os
import sys
import shutil
from glob import glob
import torch
from tqdm import tqdm

# -------- Parameters --------
NUM_VIEWS         = 60                   # 추출할 프레임 수 (각 구간별)
FRAME_RESIZE      = (1024, 1024)         # None 이면 리사이즈 안 함
VIDEO_SIZE        = (512, 512)         # 출력 비디오 해상도 (w, h)
FPS               = 30                   # 출력 비디오 fps
BUFFER_DURATION   = 1.0                  # 시작/끝 버퍼 지속 시간(초)

BG_PATH           = "grass.jpg"          # 바닥 텍스처 이미지 경로
BG_Y              = 0                    # 바닥 평면의 y 좌표
BG_LEN            = 10                    # 바닥 평면의 크기 (± BG_LEN)

ENABLE_BG_CUT     = True
OUTPUT_PATH       = "camera_orbit_multiobject_upscaled.mp4"

# ---------------------------------------------------
# **여기서 CAMERA_HEIGHT, RADIUS를 리스트로 정의합니다.**
# (각 리스트의 길이는 동일해야 합니다.)
RADIUS_LIST        = [8.0]      # 예시: 2가지 반경
CAMERA_HEIGHT_LIST = [5.0]      # 예시: 2가지 높이
# ---------------------------------------------------

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
        R_cam = np.stack([right, down, forward], axis=1).T  # 카메라 회전행렬
        t_cam = -R_cam @ C                                  # 카메라 위치를 고려한 이동벡터
        poses.append((np.concatenate([R_cam, t_cam.reshape(3,1)], axis=1), C))
    return poses

def project_point(K, extrinsic, point3D):
    """
    단일 3D 점을 투영하여 이미지 좌표(u,v)를 얻습니다.
    Z_c < 0 이면 None 반환.
    """
    pc = extrinsic[:,:3] @ point3D + extrinsic[:,3]
    proj = K @ pc
    if proj[2] <= 0:
        return None
    return (proj[:2] / proj[2]).astype(np.int32)

def get_scaled_overlay(image, distance, base_distance, base_size):
    scale = base_distance / distance
    size = int(base_size * scale)
    size = max(8, min(size, 512))
    return cv2.resize(image, (size, size)), size

def compute_plane_homography(K, extrinsic):
    """
    바닥 평면 (Y=BG_Y) 전체를 하나의 Homography로 맵핑하기 위한 행렬 H (3x3)을 계산합니다.
    평면 위의 (X, Z, 1) 동차좌표 → 이미지 상 (u', v', w') = H [X, Z, 1]^T
    """
    # extrinsic: 3x4 [R | t], 즉 세계→카메라 변환 행렬
    R = extrinsic[:,:3]
    t = extrinsic[:,3].reshape(3,1)

    # r1 = R[:,0], r3 = R[:,2], t_cam = t
    r1 = R[:,0].reshape(3,1)
    r3 = R[:,2].reshape(3,1)
    t_cam = t

    # H = K [ r1  r3  t ]
    H = K @ np.concatenate([r1, r3, t_cam], axis=1)  # shape (3,3)
    return H

def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py <dir1> <dir2> <dir3> <dir4>")
        print("── 각 dir는 객체별 프레임(이미지) 디렉토리 경로입니다.")
        sys.exit(1)

    inputs = sys.argv[1:]
    # 4개의 객체(디렉토리)로부터 각 NUM_VIEWS 장씩 이미지 로드
    frames_list = []
    for i, inp in enumerate(inputs):
        if os.path.isdir(inp):
            frames = load_images_from_dir(inp, NUM_VIEWS)
        else:
            raise RuntimeError("현재는 디렉토리 경로만 지원합니다: {}".format(inp))
        if len(frames) < NUM_VIEWS:
            raise RuntimeError(f"Not enough frames loaded for {inp}.")
        frames_list.append(frames)

    # 각 객체별 정보를 딕셔너리 리스트로 구성
    objects = [
        # Scarecrow
        {'position': np.array([0.0, BG_Y, 0.0]), 'images': frames_list[0], 'base_overlay_size': 1024},
        # Boy
        {'position': np.array([0.0, BG_Y, 1.0]), 'images': frames_list[1], 'base_overlay_size': 512},
        # # Castle (주석 처리됨)
        # {'position': np.array([-1.0, BG_Y, -1.0]), 'images': frames_list[2], 'base_overlay_size': 4000},
        # Dog
        {'position': np.array([1.0, BG_Y, -2.0]), 'images': frames_list[3], 'base_overlay_size': 512},
    ]

    # 배경 이미지 로드 및 리사이즈
    w0, h0 = VIDEO_SIZE
    bg = cv2.imread(BG_PATH, cv2.IMREAD_COLOR)
    # bg = cv2.resize(bg, (BG_LEN * 2, BG_LEN * 2), interpolation=cv2.INTER_CUBIC)

    # 카메라 Intrinsics (동일하게 사용)
    focal = 0.7 * w0
    K = np.array([[focal,   0.0, w0 / 2],
                  [  0.0, focal, h0 / 2],
                  [  0.0,   0.0,    1.0]])

    # **여러 (radius, height) 조합별로 프레임을 생성하여 master_frames에 순서대로 추가**
    master_frames = []

    # 입력된 리스트 길이 체크
    if len(CAMERA_HEIGHT_LIST) != len(RADIUS_LIST):
        raise RuntimeError("CAMERA_HEIGHT_LIST와 RADIUS_LIST는 동일한 길이여야 합니다.")

    # 이미지 좌표 전체를 한번에 계산할 그리드 (u, v)
    us = np.arange(w0)
    vs = np.arange(h0)
    grid_u, grid_v = np.meshgrid(us, vs)  # shape: (h0, w0)
    ones = np.ones_like(grid_u)           # shape: (h0, w0)

    for i, (radius, height) in enumerate(zip(RADIUS_LIST, CAMERA_HEIGHT_LIST)):
        print(f"▶ [Segment {i+1}] radius = {radius}, height = {height}")

        # (1) 카메라 궤도 포즈 생성
        poses = generate_orbit_camera_poses(radius, NUM_VIEWS, height)

        # (2) 각 뷰(view)에 대한 프레임 생성
        segment_frames = []
        for idx, (extrinsic, C) in enumerate(poses):
            # (2-1) 바닥 평면 전체를 위한 Homography H 계산
            H = compute_plane_homography(K, extrinsic)  # Y축 반전

            # (2-2) Homography 역행렬 계산 (이미지 좌표 → 평면 좌표)
            H_inv = np.linalg.inv(H)

            # (2-3) 바닥 이미지를 H로 warp
            bg_warped = cv2.warpPerspective(bg, H @ np.array([[1, 0, -BG_LEN / 2], [0, 1, -BG_LEN / 2], [0, 0, 1]]), (w0, h0))

            # (2-4) 깊이(Z_c) 테스트를 위한 마스크 생성
            #  1) 이미지 픽셀 (u, v, 1) 을 모아서 (3 x N) 배열로 만든 뒤
            #  2) H_inv 를 곱해 평면 상의 동차좌표 (X', Z', W') 계산 → 실제 (X, Z) = (X'/W', Z'/W')
            #  3) extrinsic 을 이용해 카메라 좌표계에서의 깊이 Z_c = R[2,0]*X + R[2,2]*Z + t_z 계산
            #  4) Z_c > 0 인 위치만 살아남도록 마스크

            # (u, v, 1) 모두 쌓아서 3 x (h0*w0) 배열로 만들기
            uv1 = np.stack([grid_u, grid_v, ones], axis=2).reshape(-1, 3).T  # shape: (3, N)

            # H_inv @ uv1 → plane_hom (3 x N)
            plane_hom = H_inv @ uv1  # shape: (3, N)

            # 실제 평면 좌표 (X, Z)
            X = plane_hom[0, :] / plane_hom[2, :]
            Z = plane_hom[1, :] / plane_hom[2, :]

            # extrinsic 을 그대로 사용해 깊이 Z_c 계산
            # extrinsic = [ [R00, R01, R02, t0],
            #               [R10, R11, R12, t1],
            #               [R20, R21, R22, t2] ]
            R = extrinsic[:, :3]
            t_cam = extrinsic[:, 3]
            # Z_c = R[2,0]*X + R[2,2]*Z + t_cam[2]
            Zc = R[2, 0] * X + R[2, 2] * Z + t_cam[2]

            # 깊이 조건 (Z_c > 0) 인 픽셀은 True, 나머지는 False
            mask = (Zc > 0).reshape(h0, w0)

            # (2-5) 마스크 적용: Z_c <= 0 인 픽셀은 모두 검정(0) 처리
            bg_warped[~mask] = 0

            # (2-6) 최종 바닥 프레임 생성
            frame = bg_warped.copy()

            # (2-7) 객체별 렌더링을 위한 리스트 작성 (거리순으로 depth 정렬)
            render_list = []
            for obj in objects:
                # 객체 월드 위치 → 카메라까지 거리
                dist = np.linalg.norm(C - obj['position'])
                # 객체 중심점(project_point) 투영
                p2d = project_point(K, extrinsic, obj['position'])
                if p2d is None:
                    continue

                # 객체 이미지 가져와서 적절히 스케일 조절
                overlay, sz = get_scaled_overlay(
                    obj['images'][idx],
                    dist,
                    base_distance=1.0,
                    base_size=obj['base_overlay_size']
                )
                x_img = p2d[0] - sz // 2
                y_img = p2d[1] - sz // 2

                # 화면 밖으로 나가는지 검사
                if 0 <= x_img <= w0 - sz and 0 <= y_img <= h0 - sz:
                    render_list.append((dist, overlay, sz, x_img, y_img))

            # (2-8) 알파 블렌딩: 가장 먼 것부터 가장 가까운 것 순서로 그리기
            for _, ov, sz, x_img, y_img in sorted(render_list, key=lambda x: x[0], reverse=True):
                overlay_rgb = ov[:, :, :3].astype(float)
                try:
                    alpha_mask = ov[:, :, 3].astype(float) / 255.0
                except IndexError:
                    alpha_mask = np.ones((sz, sz), dtype=float)
                alpha = cv2.merge([alpha_mask]*3)

                roi = frame[y_img:y_img+sz, x_img:x_img+sz].astype(float)
                blended = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)
                frame[y_img:y_img+sz, x_img:x_img+sz] = blended

            segment_frames.append(frame)

        # (3) 구간 앞뒤 버퍼 프레임 추가 (원하는 경우)
        buf = int(FPS * BUFFER_DURATION)
        # 만약 세그먼트 앞뒤에 버퍼를 추가하고 싶다면 아래 두 줄을 활성화:
        # segment_frames = [segment_frames[0]] * buf + segment_frames + [segment_frames[-1]] * buf

        # 마스터 프레임 리스트에 추가
        master_frames.extend(segment_frames)

    # **모든 세그먼트의 프레임을 순서대로 VideoWriter로 저장**
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (w0, h0)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter를 열 수 없습니다. 코덱(fourcc), 경로 등을 확인하세요.")

    for f in master_frames:
        writer.write(f)
    writer.release()

    print(f"✅ Video saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
