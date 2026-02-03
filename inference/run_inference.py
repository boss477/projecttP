"""
Command-line interface for surgical training video generation.

Usage:
    python inference/run_inference.py \
        --input_video surgery.mp4 \
        --output_video guided_surgery.mp4 \
        --step_model models/pretrained/step_model.pth \
        --tool_model models/pretrained/tool_model.pth
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.video_processor import (
    SurgicalVideoProcessor,
    create_demo_video,
    batch_process_videos
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate AI-powered surgical training videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python inference/run_inference.py \\
      --input_video surgery.mp4 \\
      --output_video guided_surgery.mp4 \\
      --step_model models/pretrained/step_model.pth \\
      --tool_model models/pretrained/tool_model.pth

  # Create demo video (30 seconds)
  python inference/run_inference.py \\
      --input_video surgery.mp4 \\
      --output_video demo.mp4 \\
      --demo

  # Process video segment
  python inference/run_inference.py \\
      --input_video surgery.mp4 \\
      --output_video segment.mp4 \\
      --start_time 60 \\
      --end_time 120 \\
      --step_model models/pretrained/step_model.pth \\
      --tool_model models/pretrained/tool_model.pth

  # Batch process directory
  python inference/run_inference.py \\
      --input_dir videos/ \\
      --output_dir outputs/ \\
      --batch \\
      --step_model models/pretrained/step_model.pth \\
      --tool_model models/pretrained/tool_model.pth
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input_video',
        type=str,
        help='Path to input surgical video'
    )
    parser.add_argument(
        '--output_video',
        type=str,
        help='Path to output training video'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing multiple videos (for batch mode)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save output videos'
    )
    
    # Model arguments
    parser.add_argument(
        '--step_model',
        type=str,
        help='Path to trained step recognition model'
    )
    parser.add_argument(
        '--tool_model',
        type=str,
        help='Path to trained tool detection model'
    )
    
    # Processing arguments
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for tool detection (default: 0.5)'
    )
    parser.add_argument(
        '--frame_skip',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1, all frames)'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (default: all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: auto-detect)'
    )
    
    # Segment arguments
    parser.add_argument(
        '--start_time',
        type=float,
        help='Start time in seconds (for segment processing)'
    )
    parser.add_argument(
        '--end_time',
        type=float,
        help='End time in seconds (for segment processing)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Create demo video (30 seconds)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process multiple videos'
    )
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.demo:
        # Demo mode
        if not args.input_video or not args.output_video:
            parser.error("--demo requires --input_video and --output_video")
        
        create_demo_video(
            args.input_video,
            args.output_video,
            duration=30.0
        )
        return
    
    elif args.batch:
        # Batch mode
        if not args.input_dir:
            parser.error("--batch requires --input_dir")
        if not args.step_model or not args.tool_model:
            parser.error("--batch requires --step_model and --tool_model")
        
        batch_process_videos(
            args.input_dir,
            args.output_dir,
            args.step_model,
            args.tool_model
        )
        return
    
    else:
        # Single video mode
        if not args.input_video or not args.output_video:
            parser.error("Single video mode requires --input_video and --output_video")
        if not args.step_model or not args.tool_model:
            parser.error("Single video mode requires --step_model and --tool_model")
    
    # Verify input file exists
    if not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    # Verify model files exist
    if not Path(args.step_model).exists():
        print(f"Error: Step model not found: {args.step_model}")
        sys.exit(1)
    if not Path(args.tool_model).exists():
        print(f"Error: Tool model not found: {args.tool_model}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("Surgical Training Video Generator")
    print("=" * 70)
    print(f"Input video:       {args.input_video}")
    print(f"Output video:      {args.output_video}")
    print(f"Step model:        {args.step_model}")
    print(f"Tool model:        {args.tool_model}")
    print(f"Confidence:        {args.confidence}")
    print(f"Frame skip:        {args.frame_skip}")
    print(f"Device:            {args.device or 'auto-detect'}")
    
    if args.start_time is not None and args.end_time is not None:
        print(f"Processing segment: {args.start_time}s - {args.end_time}s")
    
    print("=" * 70)
    
    # Create processor
    try:
        processor = SurgicalVideoProcessor(
            step_model_path=args.step_model,
            tool_model_path=args.tool_model,
            device=args.device,
            confidence_threshold=args.confidence
        )
    except Exception as e:
        print(f"\nError initializing processor: {e}")
        print("\nTroubleshooting:")
        print("1. Verify model files exist and are valid PyTorch checkpoints")
        print("2. Ensure CUDA is available if using GPU")
        print("3. Check that all dependencies are installed")
        sys.exit(1)
    
    # Process video
    try:
        if args.start_time is not None and args.end_time is not None:
            # Process segment
            processor.process_video_segment(
                args.input_video,
                args.output_video,
                args.start_time,
                args.end_time,
                show_progress=not args.no_progress
            )
        else:
            # Process full video
            processor.process_video(
                args.input_video,
                args.output_video,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                show_progress=not args.no_progress
            )
        
        print("\n" + "=" * 70)
        print("SUCCESS! Training video generated successfully.")
        print(f"Output saved to: {args.output_video}")
        print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print("\nTroubleshooting:")
        print("1. Verify input video is valid and readable")
        print("2. Ensure sufficient disk space for output video")
        print("3. Check GPU memory if using CUDA")
        print("4. Try processing with --frame_skip 2 or --max_frames 1000")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()