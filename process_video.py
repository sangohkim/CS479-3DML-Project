import numpy as np
import cv2
import os
import sys
import shutil
from glob import glob

# -------- Parameters --------
NUM_VIEWS = 60           # 추출할 프레임 수
UPSCALE_FACTOR = 2       # 해상도 높이기 비율
RADIUS = 3.0             # 카메라 궤도 반경
CAMERA_HEIGHT = 1      # 카메라 높이
BASE_OVERLAY_SIZE = 512   # 객체 오버레이 기본 사이즈
FPS = 30                 # 출력 비디오 fps
BUFFER_DURATION = 1.0    # 시작/끝 버퍼 지속 시간(초)
BG_PATH = "grass.jpg"  # 바닥 텍스처 이미지 경로
BG_Y = 0.0               # 바닥 평면의 y 좌표 (unused for static background)
OUTPUT_PATH = "camera_orbit_multiobject_upscaled.mp4"


def extract_frames(video_path, out_folder, max_frames, scale_factor):
    """
    비디오에서 프레임을 추출한 뒤 해상도 업스케일, PNG 저장 및 리스트 반환
    """
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] {video_path}에서 {i}번째 프레임을 읽을 수 없습니다.")
            break
        # 해상도 업스케일링
        h, w = frame.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        # PNG로 저장
        fname = os.path.join(out_folder, f"frame_{i:04d}.png")
        cv2.imwrite(fname, frame)
        frames.append(frame)
    cap.release()
    return frames


def generate_orbit_camera_poses(radius, num_views, height):
    poses = []
    for theta in np.linspace(0, 2 * np.pi, num_views, endpoint=False):
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        y = height
        cam_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        forward = (target - cam_pos)
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=1)
        T = -R.T @ cam_pos
        extrinsic = np.concatenate([R.T, T.reshape(3,1)], axis=1)
        poses.append(extrinsic)
    return poses


def project_point(K, extrinsic, point3D):
    pc = extrinsic[:,:3] @ point3D + extrinsic[:,3]
    if pc[2] <= 0:
        return None
    p  = K @ pc
    return (p[:2]/p[2]).astype(int)


def get_scaled_overlay(img, dist, base_dist, base_size):
    # 객체 크기에 업스케일 팩터 추가 반영
    scale = base_dist / dist
    sz = int(base_size * scale * UPSCALE_FACTOR)
    sz = max(8, min(sz, 512 * UPSCALE_FACTOR))
    return cv2.resize(img, (sz,sz)), sz


def remove_background_white(img):
    # 흰색 배경 제거 (threshold 기반)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,200])
    upper = np.array([180,30,255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(img, img, mask=mask_inv)
    return fg


def main():
    if len(sys.argv) != 3:
        print("사용법: python script.py <video1_path> <video2_path>")
        sys.exit(1)
    vid1, vid2 = sys.argv[1], sys.argv[2]

    # 1) 프레임 추출 및 업스케일
    frames1 = extract_frames(vid1, 'frames1', NUM_VIEWS, UPSCALE_FACTOR)
    frames2 = extract_frames(vid2, 'frames2', NUM_VIEWS, UPSCALE_FACTOR)
    if len(frames1) != NUM_VIEWS or len(frames2) != NUM_VIEWS:
        raise RuntimeError("두 비디오 모두 정확히 NUM_VIEWS개의 프레임을 가져와야 합니다.")

    # 2) 객체 정보 설정 (y 높이는 BG_Y)
    objects = [
        { 'position': np.array([0.0, BG_Y, 0.0]), 'images': frames1 },
        { 'position': np.array([1.0, BG_Y, 0.0]), 'images': frames2 }
    ]

    # 3) 배경 로드 및 업스케일 (static)
    h0, w0 = frames1[0].shape[:2]
    if not os.path.exists(BG_PATH):
        raise FileNotFoundError(f"Background not found: {BG_PATH}")
    bg = cv2.imread(BG_PATH)
    bg = cv2.resize(bg, (w0, h0), interpolation=cv2.INTER_CUBIC)

    # 4) 카메라 내부 파라미터
    focal = 0.7 * w0
    K = np.array([[focal,0,w0/2],[0,focal,h0/2],[0,0,1]])

    # 5) 궤도 카메라 포즈 생성
    poses = generate_orbit_camera_poses(RADIUS, NUM_VIEWS, CAMERA_HEIGHT)
    out_frames = []

    # 6) 각 뷰별 렌더링
    for i, pose in enumerate(poses):
        frame = bg.copy()
        render_list = []
        cam_pos = -pose[:,:3].T @ pose[:,3]
        for obj in objects:
            p3d = obj['position']; img3d = obj['images'][i]
            # 배경 제거
            img_nobg = remove_background_white(img3d)
            dist = np.linalg.norm(p3d - cam_pos)
            p2d = project_point(K, pose, p3d)
            if p2d is not None:
                overlay, sz = get_scaled_overlay(img_nobg, dist, 1.0, BASE_OVERLAY_SIZE)
                x, y = p2d - sz//2
                if 0 <= x <= w0-sz and 0 <= y <= h0-sz:
                    render_list.append((dist, overlay, sz, x, y))
        for _, ov, sz, x, y in sorted(render_list, reverse=True, key=lambda x: x[0]):
            roi = frame[y:y+sz, x:x+sz]
            frame[y:y+sz, x:x+sz] = cv2.addWeighted(roi, 0.3, ov, 0.7, 0)
        out_frames.append(frame)

    # 7) 버퍼 프레임 추가
    buf = int(FPS*BUFFER_DURATION)
    out_frames = [out_frames[0]]*buf + out_frames + [out_frames[-1]]*buf

    # 8) 비디오 작성
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0,h0))
    for f in out_frames:
        writer.write(f)
    writer.release()
    print(f"✅ Saved merged & upscaled video to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
