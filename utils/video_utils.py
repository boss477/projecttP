"""
Video I/O utilities for processing surgical videos.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from pathlib import Path


class VideoReader:
    """
    Read video frames efficiently.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def read_frames(self, frame_skip: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video.
        
        Args:
            frame_skip: Process every Nth frame (1 = all frames)
            
        Yields:
            Tuple of (frame_number, frame)
        """
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                yield frame_num, frame
            
            frame_num += 1
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by number.
        
        Args:
            frame_number: Frame index to retrieve
            
        Returns:
            Frame as numpy array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
    
    def get_properties(self) -> dict:
        """
        Get video properties as dictionary.
        
        Returns:
            Dictionary with video properties
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.frame_count / self.fps if self.fps > 0 else 0
        }


class VideoWriter:
    """
    Write video frames efficiently.
    """
    
    def __init__(
        self,
        output_path: str,
        fps: int,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height) of frames
            codec: FourCC codec (default: 'mp4v')
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not open video writer: {output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to video.
        
        Args:
            frame: Frame as numpy array (BGR format)
        """
        # Ensure frame has correct size
        if frame.shape[:2][::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
    
    def write_frames(self, frames):
        """
        Write multiple frames to video.
        
        Args:
            frames: Iterable of frames
        """
        for frame in frames:
            self.write_frame(frame)
    
    def release(self):
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video information without reading all frames.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    with VideoReader(video_path) as reader:
        return reader.get_properties()


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame to target size.
    
    Args:
        frame: Input frame
        target_size: (width, height)
        maintain_aspect: If True, maintain aspect ratio with padding
        
    Returns:
        Resized frame
    """
    if not maintain_aspect:
        return cv2.resize(frame, target_size)
    
    # Calculate scaling factor
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Add padding
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    
    return padded


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_skip: int = 30,
    prefix: str = 'frame'
) -> int:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame
        prefix: Prefix for saved frame filenames
        
    Returns:
        Number of frames extracted
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with VideoReader(video_path) as reader:
        for frame_num, frame in reader.read_frames(frame_skip):
            filename = output_path / f"{prefix}_{frame_num:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
    
    return count


def create_video_from_frames(
    frame_dir: str,
    output_path: str,
    fps: int = 30,
    pattern: str = '*.jpg'
) -> bool:
    """
    Create video from directory of frame images.
    
    Args:
        frame_dir: Directory containing frames
        output_path: Path to output video
        fps: Frames per second
        pattern: Glob pattern for frame files
        
    Returns:
        True if successful
    """
    from pathlib import Path
    import glob
    
    frame_files = sorted(glob.glob(str(Path(frame_dir) / pattern)))
    
    if not frame_files:
        return False
    
    # Read first frame to get size
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        return False
    
    h, w = first_frame.shape[:2]
    
    # Create video writer
    with VideoWriter(output_path, fps, (w, h)) as writer:
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                writer.write_frame(frame)
    
    return True


def convert_video_format(
    input_path: str,
    output_path: str,
    target_fps: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None
):
    """
    Convert video to different format/fps/size.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        target_fps: Target FPS (None = keep original)
        target_size: Target (width, height) (None = keep original)
    """
    with VideoReader(input_path) as reader:
        fps = target_fps if target_fps else reader.fps
        size = target_size if target_size else (reader.width, reader.height)
        
        with VideoWriter(output_path, fps, size) as writer:
            for _, frame in reader.read_frames():
                if target_size:
                    frame = cv2.resize(frame, target_size)
                writer.write_frame(frame)