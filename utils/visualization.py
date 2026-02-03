"""
Visualization utilities for drawing overlays and annotations on video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.constants import COLORS, FONT_SETTINGS, OVERLAY_CONFIG


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 10,
    alpha: float = 0.7
) -> np.ndarray:
    """
    Draw text with a background rectangle.
    
    Args:
        img: Input image
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font size scale
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
        alpha: Background transparency (0=transparent, 1=opaque)
        
    Returns:
        Image with text drawn
    """
    font = FONT_SETTINGS['font']
    thickness = FONT_SETTINGS['thickness']
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Blend overlay with original
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw text
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        FONT_SETTINGS['line_type']
    )
    
    return img


def draw_multiline_text(
    img: np.ndarray,
    lines: List[str],
    position: Tuple[int, int],
    font_scale: float = 0.7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    line_spacing: int = 10,
    padding: int = 15,
    alpha: float = 0.7
) -> np.ndarray:
    """
    Draw multiple lines of text with background.
    
    Args:
        img: Input image
        lines: List of text lines
        position: (x, y) starting position
        font_scale: Font size scale
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        line_spacing: Space between lines
        padding: Padding around text block
        alpha: Background transparency
        
    Returns:
        Image with text drawn
    """
    if not lines:
        return img
    
    font = FONT_SETTINGS['font']
    thickness = FONT_SETTINGS['thickness']
    
    x, y = position
    
    # Calculate total height and max width
    line_heights = []
    line_widths = []
    
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(h + baseline)
        line_widths.append(w)
    
    max_width = max(line_widths)
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    
    # Draw background
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - padding),
        (x + max_width + padding, y + total_height + padding),
        bg_color,
        -1
    )
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw each line
    current_y = y
    for line, height in zip(lines, line_heights):
        cv2.putText(
            img,
            line,
            (x, current_y + height - 5),
            font,
            font_scale,
            text_color,
            thickness,
            FONT_SETTINGS['line_type']
        )
        current_y += height + line_spacing
    
    return img


def draw_bounding_box(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box with label.
    
    Args:
        img: Input image
        bbox: (x1, y1, x2, y2) coordinates
        label: Class label
        confidence: Detection confidence
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    label_text = f"{label} [{confidence:.2f}]"
    font_scale = FONT_SETTINGS['tool_label_scale']
    
    img = draw_text_with_background(
        img,
        label_text,
        (x1, y1 - 5),
        font_scale=font_scale,
        text_color=(255, 255, 255),
        bg_color=color,
        padding=5,
        alpha=0.8
    )
    
    return img


def draw_arrow(
    img: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    tip_length: float = 0.3
) -> np.ndarray:
    """
    Draw arrow pointing to a location.
    
    Args:
        img: Input image
        start: (x, y) start point
        end: (x, y) end point
        color: Arrow color (BGR)
        thickness: Line thickness
        tip_length: Arrow tip length ratio
        
    Returns:
        Image with arrow drawn
    """
    cv2.arrowedLine(
        img,
        start,
        end,
        color,
        thickness,
        tipLength=tip_length
    )
    return img


def draw_highlight_circle(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int = 50,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 3,
    pulse: bool = False,
    frame_num: int = 0
) -> np.ndarray:
    """
    Draw a circular highlight around a point.
    
    Args:
        img: Input image
        center: (x, y) center point
        radius: Circle radius
        color: Circle color (BGR)
        thickness: Line thickness
        pulse: If True, create pulsing effect
        frame_num: Current frame number (for pulse effect)
        
    Returns:
        Image with circle drawn
    """
    if pulse:
        # Create pulsing effect
        pulse_factor = 1.0 + 0.3 * np.sin(frame_num * 0.2)
        radius = int(radius * pulse_factor)
    
    cv2.circle(img, center, radius, color, thickness)
    return img


def create_step_overlay_bar(
    img: np.ndarray,
    step_num: int,
    step_name: str,
    total_steps: int = 7
) -> np.ndarray:
    """
    Create top overlay bar showing current step.
    
    Args:
        img: Input image
        step_num: Current step number (0-indexed)
        step_name: Name of current step
        total_steps: Total number of steps
        
    Returns:
        Image with step overlay
    """
    h, w = img.shape[:2]
    bar_height = OVERLAY_CONFIG['step_bar_height']
    
    # Create overlay
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (w, bar_height),
        COLORS['step_bg'],
        -1
    )
    
    # Blend
    alpha = OVERLAY_CONFIG['alpha_step']
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Add text
    text = f"Step {step_num + 1}/{total_steps}: {step_name}"
    font_scale = FONT_SETTINGS['step_scale']
    thickness = FONT_SETTINGS['thickness']
    font = FONT_SETTINGS['font']
    
    # Center text
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2
    
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        COLORS['step_text'],
        thickness,
        FONT_SETTINGS['line_type']
    )
    
    return img


def create_instruction_overlay(
    img: np.ndarray,
    instructions: List[str],
    position: str = 'bottom-left'
) -> np.ndarray:
    """
    Create instruction text overlay box.
    
    Args:
        img: Input image
        instructions: List of instruction strings
        position: Position ('bottom-left', 'bottom-right', 'top-left', 'top-right')
        
    Returns:
        Image with instructions
    """
    h, w = img.shape[:2]
    padding = OVERLAY_CONFIG['padding']
    
    # Determine position
    if position == 'bottom-left':
        x, y = padding, h - OVERLAY_CONFIG['instruction_box_height'] - padding
    elif position == 'bottom-right':
        x = w - OVERLAY_CONFIG['instruction_box_width'] - padding
        y = h - OVERLAY_CONFIG['instruction_box_height'] - padding
    elif position == 'top-left':
        x, y = padding, OVERLAY_CONFIG['step_bar_height'] + padding
    else:  # top-right
        x = w - OVERLAY_CONFIG['instruction_box_width'] - padding
        y = OVERLAY_CONFIG['step_bar_height'] + padding
    
    # Add title
    title = "DO THIS NOW:"
    lines = [title, ""] + instructions[:3]  # Show up to 3 instructions
    
    img = draw_multiline_text(
        img,
        lines,
        (x, y),
        font_scale=FONT_SETTINGS['instruction_scale'],
        text_color=COLORS['instruction_text'],
        bg_color=COLORS['instruction_bg'],
        line_spacing=8,
        padding=15,
        alpha=OVERLAY_CONFIG['alpha_instruction']
    )
    
    return img


def add_visual_guidance(
    img: np.ndarray,
    tool_boxes: List[Tuple[int, int, int, int]],
    attention_points: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """
    Add visual guidance arrows and highlights.
    
    Args:
        img: Input image
        tool_boxes: List of tool bounding boxes
        attention_points: Optional list of points to highlight
        
    Returns:
        Image with visual guidance
    """
    h, w = img.shape[:2]
    
    # Add arrows pointing to tools
    for bbox in tool_boxes:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Arrow from top
        arrow_start = (center_x, max(y1 - 100, 50))
        arrow_end = (center_x, y1 - 10)
        
        img = draw_arrow(img, arrow_start, arrow_end, COLORS['arrow'], thickness=3)
    
    # Add highlight circles to attention points
    if attention_points:
        for i, point in enumerate(attention_points):
            img = draw_highlight_circle(
                img,
                point,
                radius=60,
                color=COLORS['highlight'],
                thickness=3,
                pulse=True,
                frame_num=i
            )
    
    return img


def create_progress_bar(
    img: np.ndarray,
    current_step: int,
    total_steps: int,
    position: str = 'bottom'
) -> np.ndarray:
    """
    Create a progress bar showing procedure progress.
    
    Args:
        img: Input image
        current_step: Current step (0-indexed)
        total_steps: Total number of steps
        position: 'top' or 'bottom'
        
    Returns:
        Image with progress bar
    """
    h, w = img.shape[:2]
    bar_height = 20
    bar_width = w - 100
    padding = 50
    
    # Position
    x = padding
    y = h - padding if position == 'bottom' else padding + OVERLAY_CONFIG['step_bar_height']
    
    # Background bar
    cv2.rectangle(
        img,
        (x, y),
        (x + bar_width, y + bar_height),
        (100, 100, 100),
        -1
    )
    
    # Progress bar
    progress = (current_step + 1) / total_steps
    progress_width = int(bar_width * progress)
    
    cv2.rectangle(
        img,
        (x, y),
        (x + progress_width, y + bar_height),
        (0, 255, 0),
        -1
    )
    
    # Border
    cv2.rectangle(
        img,
        (x, y),
        (x + bar_width, y + bar_height),
        (255, 255, 255),
        2
    )
    
    return img