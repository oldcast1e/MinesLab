# 맥북 기준 소스 코드

import cv2, glob, numpy as np, os, json

# 경로 설정 (항상 프로젝트 루트 기준)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)               # 상위: IGEV-plusplus/
LEFT_DIR  = os.path.join(PROJECT_ROOT, "calib/left/*.png")
RIGHT_DIR = os.path.join(PROJECT_ROOT, "calib/right/*.png")
OUT_DIR   = os.path.join(PROJECT_ROOT, "calib_out")
PATTERN   = (9, 6)        # 체커보드 내부 코너 수
SQUARE_MM = 25.0          # 한 칸 크기(mm)

os.makedirs(OUT_DIR, exist_ok=True)

# 코너 월드 좌표
objp = np.zeros((PATTERN[0]*PATTERN[1],3), np.float32)
objp[:,:2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1,2)
objp *= SQUARE_MM

objpoints, imgpoints_l, imgpoints_r = [], [], []
left_paths  = sorted(glob.glob(LEFT_DIR))
right_paths = sorted(glob.glob(RIGHT_DIR))

for lp, rp in zip(left_paths, right_paths):
    imgL, imgR = cv2.imread(lp, cv2.IMREAD_GRAYSCALE), cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    retL, cornersL = cv2.findChessboardCorners(imgL, PATTERN)
    retR, cornersR = cv2.findChessboardCorners(imgR, PATTERN)
    if retL and retR:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), term)
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), term)
        objpoints.append(objp)
        imgpoints_l.append(cornersL)
        imgpoints_r.append(cornersR)

# 카메라 보정
img_shape = imgL.shape[::-1]
_, KL, DL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
_, KR, DR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

# 스테레오 보정
flags = cv2.CALIB_FIX_INTRINSIC
_, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    KL, DL, KR, DR, img_shape,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=flags
)

# 레티파이
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(KL, DL, KR, DR, img_shape, R, T, flags=0)
mapLx, mapLy = cv2.initUndistortRectifyMap(KL, DL, RL, PL, img_shape, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(KR, DR, RR, PR, img_shape, cv2.CV_32FC1)

# 결과 저장
np.save(os.path.join(OUT_DIR, "mapLx.npy"), mapLx)
np.save(os.path.join(OUT_DIR, "mapLy.npy"), mapLy)
np.save(os.path.join(OUT_DIR, "mapRx.npy"), mapRx)
np.save(os.path.join(OUT_DIR, "mapRy.npy"), mapRy)

np.save(os.path.join(OUT_DIR, "KL.npy"), KL)
np.save(os.path.join(OUT_DIR, "KR.npy"), KR)
np.save(os.path.join(OUT_DIR, "PL.npy"), PL)
np.save(os.path.join(OUT_DIR, "PR.npy"), PR)
np.save(os.path.join(OUT_DIR, "R.npy"), R)
np.save(os.path.join(OUT_DIR, "T.npy"), T)

fx_pixel = float(PL[0,0])
baseline_m = float(abs(T[0,0]) / 1000.0)  # mm → m

info = {
    "fx_pixel": fx_pixel,
    "baseline_m_from_T": baseline_m
}
with open(os.path.join(OUT_DIR, "info.json"), "w") as f:
    json.dump(info, f, indent=2)

print("====================================")
print(f"fx (pixel)        = {fx_pixel}")
print(f"T vector (mm)     = {T.ravel()}")
print(f"추정 baseline (m) = {baseline_m}")
print(f"결과가 {OUT_DIR}/ 폴더에 저장되었습니다.")
print("====================================")
