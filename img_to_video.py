import cv2
import os

# 設置資料夾路徑和輸出視頻檔案名稱
root_path = 'test_similar_img_t10'
output_video = "test_video/video_similar_t10.mp4"  # Add file extension to the output video

# 獲取第一張圖片的寬度和高度，用於設置視頻的寬度和高度
first_image = cv2.imread(f"{root_path}/1.png")  # Corrected file path
frame_width = first_image.shape[1]
frame_height = first_image.shape[0]

# 創建VideoWriter對象來寫入視頻
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編解碼器
out = cv2.VideoWriter(output_video, fourcc, 15.0, (frame_width, frame_height))

# 逐一讀取PNG圖片，並將它們寫入視頻
for i in range(1, 101):
    frame = cv2.imread(f"{root_path}/{i}.png")  # Use variable i to iterate through images
    out.write(frame)

# 釋放VideoWriter對象
out.release()

print(f'視頻已經生成: {output_video}')
