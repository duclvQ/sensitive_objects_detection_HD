import cv2
import timeit
def display_video(video_path):
    # Mở video để đọc
    cap = cv2.VideoCapture(video_path)
    # Set the desired FPS
    #cap.set(cv2.CAP_PROP_FPS, 5)
    # Kiểm tra xem video có được mở thành công hay không
    if not cap.isOpened():
        print("Không thể mở video.")
        return
    start = timeit.default_timer()
    frame_num = 0
    while True:
        # Đọc một khung hình từ video
        # Set the starting frame position
        
        # s = timeit.default_timer()
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # frame_num += 10
        ret, frame = cap.read()
        e = timeit.default_timer()
        frame_num+=1
        # print(frame_num)
        # Kiểm tra xem việc đọc khung hình có thành công hay không
        if not ret or frame_num >=10000000:
            break
        
        # print('1f:', e-s)
        # Hiển thị khung hình
        #cv2.imshow('Video', frame)

        ## Đợi một khoảng thời gian (đơn vị mili giây) và kiểm tra xem có phải là phím 'q' không
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    stop = timeit.default_timer()
    print(stop - start)
    # Giải phóng các tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để đọc và hiển thị video
video_path = '22.mp4'  # Thay đổi đường dẫn tới video của bạn
display_video(video_path)
