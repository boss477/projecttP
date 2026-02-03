import cv2
import os

def re_encode_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try different codecs
    codecs = ['avc1', 'H264', 'mp4v']
    success = False
    
    for codec in codecs:
        print(f"Trying codec: {codec}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            print(f"Codec {codec} is supported.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            out.release()
            success = True
            break
        else:
            print(f"Codec {codec} is NOT supported.")
            
    cap.release()
    if success:
        print(f"Video re-encoded successfully to {output_path}")
    else:
        print("Failed to re-encode video with any available codec.")

if __name__ == "__main__":
    input_vid = "outputs/demo_result.mp4"
    output_vid = "outputs/demo_result_h264.mp4"
    re_encode_video(input_vid, output_vid)
