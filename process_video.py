import numpy as np
import cv2
import os
import sys
import shutil
from glob import glob
import torch
from carvekit.api.high import HiInterface

# -------- Parameters --------
NUM_VIEWS = 60           # 추출할 프레임 수
UPSCALE_FACTOR = 2       # 해상도 업스케일 비율
RADIUS = 15.0            # 카메라 궤도 반경
CAMERA_HEIGHT = 5     # 카메라 높이 (world y)
BASE_OVERLAY_SIZE = 512   # 객체 오버레이 기본 사이즈
FPS = 30                 # 출력 비디오 fps
BUFFER_DURATION = 1.0    # 시작/끝 버퍼 지속 시간(초)
BG_PATH = "grass.jpg"   # 바닥 텍스처 이미지 경로
BG_Y = 0               # 바닥 평면의 y 좌표
BG_LEN = 8.0
OUTPUT_PATH = "camera_orbit_multiobject_upscaled.mp4"


def extract_frames(video_path, out_folder, max_frames, scale_factor):
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    
    if os.path.exists(out_folder): shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Warning] {video_path} frame {i} unreadable.")
            break
        # 해상도 업스케일
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_CUBIC)
        frame_path = os.path.join(out_folder, f"frame_{i:04d}.png")
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        frames.append(frame)
    cap.release()
    frames_wo_bg = interface(frame_paths)
    frames = []
    for i, frame_wo_bg in enumerate(frames_wo_bg):
        if frame_wo_bg is not None:
            frame_wo_bg.save(frame_paths[i])
            frames.append(cv2.imread(frame_paths[i]))
        else:
            print(f"[Warning] {video_path} frame {i} segmentation failed.")
            raise RuntimeError(f"Frame {i} segmentation failed.")
    return frames


def generate_orbit_camera_poses(radius, num_views, height):
    poses = []
    for theta in np.linspace(0, 2*np.pi, num_views, endpoint=False):
        # 카메라 위치
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        C = np.array([x, height, z])
        # 바라볼 대상
        target = np.array([0.0, 0.0, 0.0])
        forward = (target - C)
        forward /= np.linalg.norm(forward)
        up_vec = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_vec, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        # world->camera rotation and translation
        R_cam = np.stack([right, up, forward], axis=1).T
        t_cam = -R_cam @ C
        extrinsic = np.concatenate([R_cam, t_cam.reshape(3,1)], axis=1)
        poses.append((extrinsic, C))
    return poses


def project_point(K, extrinsic, point3D):
    pc = extrinsic[:,:3] @ point3D + extrinsic[:,3]
    # if pc[2] <= 0:
    #     return None
    proj = K @ pc
    return (proj[:2]/proj[2]).astype(np.int32)


def get_scaled_overlay(img, dist, base_dist, base_size):
    sz = int(base_size * (base_dist/dist) * UPSCALE_FACTOR)
    sz = max(8, min(sz, 512*UPSCALE_FACTOR))
    return cv2.resize(img, (sz,sz), interpolation=cv2.INTER_CUBIC), sz


def compute_ground_homography(K, extrinsic, bg_size, w, h):
    # 바닥 평면의 4개 모서리 점 정의 (큰 영역)
    ground_points = np.array([
        [-bg_size, BG_Y, -bg_size],  # 좌상단
        [bg_size, BG_Y, -bg_size],   # 우상단  
        [bg_size, BG_Y, bg_size],    # 우하단
        [-bg_size, BG_Y, bg_size]    # 좌하단
    ])
    
    # 이미지 평면으로 투영
    image_points = []
    visible_points = []
    for i, pt in enumerate(ground_points):
        proj = project_point(K, extrinsic, pt)
        if proj is not None:
            image_points.append(proj)
            visible_points.append(i)
        else:
            # 카메라 뒤에 있는 점 처리
            image_points.append(None)
    
    # 모든 점이 카메라 뒤에 있는 경우
    if len(visible_points) < 2:
        raise RuntimeError("Not enough visible points to compute homography.")
    
    # # 카메라 뒤에 있는 점들에 대한 추정
    # if len(visible_points) == 4:
    #     image_points = sorted(image_points, key=lambda x: (x[1], x[0]))  # y좌표로 정렬
    # elif len(visible_points) == 3:
    #     print("This case.")
        
    # elif len(visible_points) == 2:
    #     image_points = sorted(image_points, key=lambda x: (x[0], x[1]))  # y좌표로 정렬
    #     image_points = image_points + [[0, h], [w, h]]  # 나머지 두 점은 바닥의 오른쪽 아래 모서리로 채움
    image_points = sorted(image_points, key=lambda x: (x[1], x[0]))  # y좌표로 정렬
    
    # 점들 배열로 변환
    image_points = np.array(image_points, dtype=np.float32)
    
    # 좌표 정렬 (왼쪽 위부터 시계방향)
    tl = image_points[0]
    tr = image_points[1]
    bl = image_points[2]
    br = image_points[3]
    
    # 목표 이미지 모서리 점
    dst_points = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)
    
    # 호모그래피 계산
    src_points = np.array([tl, tr, br, bl], dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(dst_points, src_points)
        return H
    except:
        print("[Warning] Failed to compute homography. Using fallback method.")
        
        # 대안적 방법: 단순 어파인 변환 (3점만 필요)
        if len(visible_points) >= 3:
            visible_src = np.array([image_points[i] for i in visible_points[:3]], dtype=np.float32)
            visible_dst = np.array([dst_points[i] for i in visible_points[:3]], dtype=np.float32)
            M = cv2.getAffineTransform(visible_dst, visible_src)
            # 3x3 호모그래피 형태로 변환
            H = np.vstack([M, [0, 0, 1]])
            return H
        
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <video1> <video2>")
        sys.exit(1)
    vid1, vid2 = sys.argv[1], sys.argv[2]

    # 프레임 추출 및 업스케일
    frames1 = extract_frames(vid1, 'frames1', NUM_VIEWS, UPSCALE_FACTOR)
    frames2 = extract_frames(vid2, 'frames2', NUM_VIEWS, UPSCALE_FACTOR)

    # 객체 설정 (y=BG_Y in xz-plane)
    objects = [
        {'position': np.array([0.0, BG_Y, 0.0]), 'images': frames1},
        {'position': np.array([2.0, BG_Y, 0.0]), 'images': frames2}
    ]

    # 배경 로드 및 업스케일
    h0, w0 = frames1[0].shape[:2]
    bg = cv2.imread(BG_PATH)
    bg = cv2.resize(bg, (w0, h0), interpolation=cv2.INTER_CUBIC)

    # 카메라 내부 파라미터
    focal = 0.7 * w0
    K = np.array([[focal,0,w0/2],[0,focal,h0/2],[0,0,1]])

    # 궤도 포즈 생성
    poses = generate_orbit_camera_poses(RADIUS, NUM_VIEWS, CAMERA_HEIGHT)
    out_frames = []

    # 각 뷰 렌더링
    for extrinsic, C in poses:
        H = compute_ground_homography(K, extrinsic, BG_LEN, *frames1[0].shape[:2])  # 바닥 크기
    
        if H is not None:
            bg_warped = cv2.warpPerspective(bg, H, (w0, h0))
        else:
            # 호모그래피 계산 실패시 기본 배경 사용
            raise RuntimeError("Failed to compute homography for background warping.")
            bg_warped = cv2.resize(bg, (w0, h0))

        frame = bg_warped.copy()
        frame = cv2.flip(frame, -1)

        # 객체 합성
        render_list = []
        for obj in objects:
            p3d = obj['position']; img3d = obj['images'][len(out_frames)]
            dist = np.linalg.norm(C - p3d)
            p2d = project_point(K, extrinsic, p3d)
            if p2d is not None:
                overlay, sz = get_scaled_overlay(img3d, dist, 1.0, BASE_OVERLAY_SIZE)
                x,y = p2d - sz//2
                if 0<=x<=w0-sz and 0<=y<=h0-sz:
                    render_list.append((dist, overlay, sz, x, y))
            else:
                # print(f"[Warning] Object at {p3d} not visible in view {len(out_frames)}")
                ...
        for _, ov, sz, x, y in sorted(render_list, reverse=True, key=lambda x: x[0]):
            roi = frame[y:y+sz, x:x+sz]
            frame[y:y+sz, x:x+sz] = cv2.addWeighted(roi,0.3,ov,0.7,0)
        out_frames.append(frame)

    # 버퍼 프레임
    buf = int(FPS*BUFFER_DURATION)
    frames_buf = [out_frames[0]]*buf + out_frames + [out_frames[-1]]*buf

    # 비디오 작성
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w0, h0))
    for f in frames_buf:
        writer.write(f)
    writer.release()
    print(f"✅ Saved merged video: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
