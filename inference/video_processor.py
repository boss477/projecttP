"""
Main video processor for surgical training system.
Combines step recognition, tool detection, and overlay rendering.
"""

import cv2
import torch
import numpy as np
from typing import Optional, Callable
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.step_recognition import load_step_model, StepRecognitionInference
from models.tool_detection import load_tool_model, ToolDetectionModel
from utils.video_utils import VideoReader, VideoWriter
from inference.overlay_renderer import OverlayRenderer


class SurgicalVideoProcessor:
    """
    Main processor for generating guided training videos.
    """
    
    def __init__(
        self,
        step_model_path: str,
        tool_model_path: str,
        device: str = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize video processor.
        
        Args:
            step_model_path: Path to trained step recognition model
            tool_model_path: Path to trained tool detection model
            device: Device to run inference on
            confidence_threshold: Confidence threshold for tool detection
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Initializing Surgical Video Processor on {self.device}...")
        
        # Load models
        print("Loading step recognition model...")
        self.step_model = load_step_model(step_model_path, self.device)
        
        print("Loading tool detection model...")
        self.tool_model = load_tool_model(
            tool_model_path,
            confidence_threshold=confidence_threshold,
            device=self.device
        )
        
        # Create overlay renderer
        self.renderer = OverlayRenderer(
            show_step_bar=True,
            show_instructions=True,
            show_tool_boxes=True,
            show_visual_guidance=True,
            show_progress_bar=False
        )
        
        print("Initialization complete!")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int = 0
    ) -> tuple:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR)
            frame_num: Frame number for tracking
            
        Returns:
            Tuple of (processed_frame, step_prediction, tool_detections)
        """
        # Prepare frame for step recognition (RGB, 224x224)
        step_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        step_input = cv2.resize(step_input, (224, 224))
        step_input = torch.from_numpy(step_input).permute(2, 0, 1)
        
        # Predict step
        step_prediction = self.step_model.predict(step_input, use_temporal=True)
        
        # Detect tools
        tool_detections = self.tool_model.predict(frame, return_boxes=True)
        
        # Render overlays
        processed_frame = self.renderer.render(
            frame,
            step_prediction,
            tool_detections,
            frame_num
        )
        
        return processed_frame, step_prediction, tool_detections
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        """
        Process entire video and generate training video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            frame_skip: Process every Nth frame (1 = all frames)
            max_frames: Maximum frames to process (None = all)
            show_progress: Show progress bar
            progress_callback: Optional callback function for progress updates
        """
        print(f"\nProcessing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Reset step model state
        self.step_model.reset()
        
        # Open input video
        with VideoReader(input_path) as reader:
            # Get video properties
            props = reader.get_properties()
            print(f"Video properties:")
            print(f"  - Resolution: {props['width']}x{props['height']}")
            print(f"  - FPS: {props['fps']}")
            print(f"  - Frames: {props['frame_count']}")
            print(f"  - Duration: {props['duration_seconds']:.1f}s")
            
            # Determine number of frames to process
            total_frames = props['frame_count']
            if max_frames:
                total_frames = min(total_frames, max_frames)
            
            # Create output video writer
            with VideoWriter(
                output_path,
                fps=props['fps'],
                frame_size=(props['width'], props['height'])
            ) as writer:
                
                # Process frames
                frame_count = 0
                processed_count = 0
                
                iterator = reader.read_frames(frame_skip)
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        total=total_frames // frame_skip,
                        desc="Processing frames"
                    )
                
                for frame_num, frame in iterator:
                    if max_frames and processed_count >= max_frames:
                        break
                    
                    # Process frame
                    try:
                        processed_frame, step_pred, tool_dets = self.process_frame(
                            frame,
                            frame_num
                        )
                        
                        # Write to output
                        writer.write_frame(processed_frame)
                        
                        processed_count += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(processed_count, total_frames)
                    
                    except Exception as e:
                        print(f"\nError processing frame {frame_num}: {e}")
                        # Write original frame on error
                        writer.write_frame(frame)
                    
                    frame_count += 1
        
        print(f"\nProcessing complete!")
        print(f"Processed {processed_count} frames")
        print(f"Output saved to: {output_path}")
    
    def process_video_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        show_progress: bool = True
    ):
        """
        Process a segment of video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            start_time: Start time in seconds
            end_time: End time in seconds
            show_progress: Show progress bar
        """
        with VideoReader(input_path) as reader:
            props = reader.get_properties()
            fps = props['fps']
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            print(f"Processing segment: {start_time}s - {end_time}s")
            print(f"Frames: {start_frame} - {end_frame}")
            
            # Process segment
            self.process_video(
                input_path,
                output_path,
                frame_skip=1,
                max_frames=end_frame - start_frame,
                show_progress=show_progress
            )


def create_demo_video(
    input_path: str,
    output_path: str,
    duration: float = 30.0
):
    """
    Create a short demo video for testing.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        duration: Duration in seconds
    """
    print(f"Creating demo video ({duration}s)...")
    
    # For demo, use dummy models if real ones not available
    try:
        processor = SurgicalVideoProcessor(
            step_model_path='models/pretrained/step_model.pth',
            tool_model_path='models/pretrained/tool_model.pth'
        )
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Creating simple demo without AI processing...")
        
        # Simple demo without models
        with VideoReader(input_path) as reader:
            props = reader.get_properties()
            max_frames = int(duration * props['fps'])
            
            with VideoWriter(
                output_path,
                fps=props['fps'],
                frame_size=(props['width'], props['height'])
            ) as writer:
                
                for frame_num, frame in tqdm(
                    reader.read_frames(),
                    total=max_frames,
                    desc="Creating demo"
                ):
                    if frame_num >= max_frames:
                        break
                    
                    # Add simple text overlay
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
                    cv2.putText(
                        frame,
                        "AI-Powered Surgical Training - Demo",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2
                    )
                    
                    writer.write_frame(frame)
        
        print(f"Demo video created: {output_path}")
        return
    
    # Process with full pipeline
    processor.process_video_segment(
        input_path,
        output_path,
        start_time=0.0,
        end_time=duration,
        show_progress=True
    )
    
    print(f"Demo video created: {output_path}")


def batch_process_videos(
    input_dir: str,
    output_dir: str,
    step_model_path: str,
    tool_model_path: str,
    pattern: str = '*.mp4'
):
    """
    Process multiple videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output videos
        step_model_path: Path to step model
        tool_model_path: Path to tool model
        pattern: Glob pattern for video files
    """
    from pathlib import Path
    import glob
    
    # Get video files
    video_files = glob.glob(str(Path(input_dir) / pattern))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create processor
    processor = SurgicalVideoProcessor(
        step_model_path=step_model_path,
        tool_model_path=tool_model_path
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for video_file in video_files:
        video_name = Path(video_file).stem
        output_file = Path(output_dir) / f"{video_name}_guided.mp4"
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        
        try:
            processor.process_video(
                str(video_file),
                str(output_file),
                show_progress=True
            )
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Batch processing complete!")