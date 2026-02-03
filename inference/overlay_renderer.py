"""
Overlay renderer for adding instructional overlays to surgical video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import (
    SURGICAL_STEPS,
    STEP_INSTRUCTIONS,
    TOOL_CLASSES,
    COLORS,
    TOOL_RECOMMENDATIONS
)
from utils.visualization import (
    create_step_overlay_bar,
    create_instruction_overlay,
    draw_bounding_box,
    draw_arrow,
    draw_highlight_circle,
    create_progress_bar
)


class OverlayRenderer:
    """
    Renders instructional overlays on video frames.
    """
    
    def __init__(
        self,
        show_step_bar: bool = True,
        show_instructions: bool = True,
        show_tool_boxes: bool = True,
        show_visual_guidance: bool = True,
        show_progress_bar: bool = False
    ):
        """
        Initialize overlay renderer.
        
        Args:
            show_step_bar: Show step name bar at top
            show_instructions: Show instructional text box
            show_tool_boxes: Show tool bounding boxes
            show_visual_guidance: Show arrows and highlights
            show_progress_bar: Show progress bar
        """
        self.show_step_bar = show_step_bar
        self.show_instructions = show_instructions
        self.show_tool_boxes = show_tool_boxes
        self.show_visual_guidance = show_visual_guidance
        self.show_progress_bar = show_progress_bar
        
        self.frame_count = 0
    
    def render(
        self,
        frame: np.ndarray,
        step_prediction: int,
        tool_detections: List[Dict],
        frame_num: int = 0
    ) -> np.ndarray:
        """
        Render all overlays on a frame.
        
        Args:
            frame: Input video frame (BGR)
            step_prediction: Predicted surgical step (0-6)
            tool_detections: List of tool detections
            frame_num: Current frame number (for animations)
            
        Returns:
            Frame with overlays rendered
        """
        result = frame.copy()
        
        # 1. Step overlay bar at top
        if self.show_step_bar:
            result = self._render_step_bar(result, step_prediction)
        
        # 2. Tool bounding boxes
        if self.show_tool_boxes and tool_detections:
            result = self._render_tool_boxes(result, tool_detections)
        
        # 3. Visual guidance (arrows, highlights)
        if self.show_visual_guidance and tool_detections:
            result = self._render_visual_guidance(
                result,
                tool_detections,
                step_prediction,
                frame_num
            )
        
        # 4. Instructional text overlay
        if self.show_instructions:
            result = self._render_instructions(result, step_prediction)
        
        # 5. Progress bar
        if self.show_progress_bar:
            result = create_progress_bar(result, step_prediction, 7, position='bottom')
        
        self.frame_count += 1
        
        return result
    
    def _render_step_bar(
        self,
        frame: np.ndarray,
        step_num: int
    ) -> np.ndarray:
        """Render step name bar at top of frame."""
        step_name = SURGICAL_STEPS.get(step_num, "Unknown Step")
        return create_step_overlay_bar(frame, step_num, step_name, total_steps=7)
    
    def _render_tool_boxes(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Render bounding boxes around detected tools."""
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            bbox = det['bbox']
            
            # Get color for this tool
            color = COLORS['tool_colors'].get(class_name, (0, 255, 0))
            
            # Draw bounding box
            frame = draw_bounding_box(
                frame,
                bbox,
                class_name,
                confidence,
                color=color,
                thickness=3
            )
        
        return frame
    
    def _render_visual_guidance(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        step_num: int,
        frame_num: int
    ) -> np.ndarray:
        """Render visual guidance arrows and highlights."""
        h, w = frame.shape[:2]
        
        # Get recommended tools for current step
        recommended_tools = TOOL_RECOMMENDATIONS.get(step_num, [])
        
        for det in detections:
            class_name = det['class_name']
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if this tool is recommended for current step
            is_recommended = class_name in recommended_tools
            
            if is_recommended:
                # Draw arrow pointing to recommended tool
                arrow_start = (center_x, max(y1 - 80, 50))
                arrow_end = (center_x, y1 - 10)
                
                frame = draw_arrow(
                    frame,
                    arrow_start,
                    arrow_end,
                    color=COLORS['arrow'],
                    thickness=4
                )
                
                # Add pulsing highlight
                frame = draw_highlight_circle(
                    frame,
                    (center_x, center_y),
                    radius=70,
                    color=COLORS['highlight'],
                    thickness=3,
                    pulse=True,
                    frame_num=frame_num
                )
        
        return frame
    
    def _render_instructions(
        self,
        frame: np.ndarray,
        step_num: int
    ) -> np.ndarray:
        """Render instructional text overlay."""
        # Get instructions for current step
        instructions = STEP_INSTRUCTIONS.get(step_num, ["No instructions available"])
        
        # Show first 3 instructions
        display_instructions = instructions[:3]
        
        # Add overlay
        frame = create_instruction_overlay(
            frame,
            display_instructions,
            position='bottom-left'
        )
        
        return frame
    
    def render_batch(
        self,
        frames: List[np.ndarray],
        step_predictions: List[int],
        tool_detections_list: List[List[Dict]]
    ) -> List[np.ndarray]:
        """
        Render overlays on a batch of frames.
        
        Args:
            frames: List of video frames
            step_predictions: List of step predictions
            tool_detections_list: List of tool detections for each frame
            
        Returns:
            List of frames with overlays
        """
        rendered_frames = []
        
        for i, (frame, step, detections) in enumerate(
            zip(frames, step_predictions, tool_detections_list)
        ):
            rendered = self.render(frame, step, detections, frame_num=i)
            rendered_frames.append(rendered)
        
        return rendered_frames


class MinimalOverlayRenderer:
    """
    Simplified overlay renderer with minimal visual elements.
    """
    
    def __init__(self):
        """Initialize minimal renderer."""
        pass
    
    def render(
        self,
        frame: np.ndarray,
        step_name: str,
        instruction: str,
        tools: List[str]
    ) -> np.ndarray:
        """
        Render minimal overlay.
        
        Args:
            frame: Input frame
            step_name: Current step name
            instruction: Single instruction text
            tools: List of detected tool names
            
        Returns:
            Frame with minimal overlay
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Top bar - step name
        cv2.rectangle(result, (0, 0), (w, 50), (50, 50, 50), -1)
        cv2.putText(
            result,
            step_name,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Bottom bar - instruction
        cv2.rectangle(result, (0, h-80), (w, h), (50, 50, 50), -1)
        
        # Wrap text if too long
        if len(instruction) > 80:
            words = instruction.split()
            line1 = ' '.join(words[:len(words)//2])
            line2 = ' '.join(words[len(words)//2:])
            
            cv2.putText(
                result,
                line1,
                (20, h-55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.putText(
                result,
                line2,
                (20, h-25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                result,
                instruction,
                (20, h-40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        # Tool names in corner
        if tools:
            tools_text = "Tools: " + ", ".join(tools[:3])
            cv2.putText(
                result,
                tools_text,
                (w-400, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return result


def create_demo_overlay(
    frame: np.ndarray,
    demo_text: str = "AI-Powered Surgical Training System"
) -> np.ndarray:
    """
    Create a demo watermark overlay.
    
    Args:
        frame: Input frame
        demo_text: Demo text to display
        
    Returns:
        Frame with demo overlay
    """
    result = frame.copy()
    h, w = result.shape[:2]
    
    # Semi-transparent watermark
    overlay = result.copy()
    cv2.putText(
        overlay,
        demo_text,
        (w//2 - 300, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    return result