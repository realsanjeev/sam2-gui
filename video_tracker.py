import argparse
import os
import logging
import subprocess
import warnings
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np

import torch
import colorsys
from sam2.build_sam import build_sam2_video_predictor


warnings.filterwarnings("ignore")
# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PromptGUI:
    def __init__(self, checkpoint_dir: str, model_cfg: str, device: str):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.device = device

        # Core state
        self.sam_model = None
        self.inference_state = None

        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        # Image directory
        self.img_dir: str = ""
        self.img_paths: list[str] = []
        self._init_sam_model()

    # -------------------------------------------------------------
    # Initialization & reset
    # -------------------------------------------------------------
    def _init_sam_model(self) -> None:
        """Load the SAM model if it hasn't been initialized yet."""
        if self.sam_model is None:
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir, device=self.device)
            logger.info(f"Loaded model checkpoint from {self.checkpoint_dir}")

    def _clear_image(self) -> None:
        """Reset current image and all related mask/logit data."""
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks.clear()
        self.cur_logits.clear()
        self.index_masks_all.clear()
        self.color_masks_all.clear()

    def reset(self) -> None:
        """Completely reset image data and inference state."""
        self._clear_image()
        if self.inference_state is not None:
            self.sam_model.reset_state(self.inference_state)

    # -------------------------------------------------------------
    # Image management
    # -------------------------------------------------------------
    def set_img_dir(self, img_dir: str) -> int:
        """Load and store all image paths from the given directory."""
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if is_image(f)
        ]
        
        return len(self.img_paths)

    def set_input_image(self, index: int = 0) -> Optional[np.ndarray]:
        """Set the current image by index from loaded paths."""
        logger.info(f"Setting frame {index} / {len(self.img_paths)}")
        if index < 0 or index >= len(self.img_paths):
            logger.warning("Requested frame index out of range")
            return self.image
        self.clear_points()
        self.frame_index = index
        self.image = iio.imread(self.img_paths[index])
        return self.image

    # -------------------------------------------------------------
    # Point and mask management
    # -------------------------------------------------------------
    def clear_points(self) -> Tuple[None, None, str]:
        """Clear all selected points and labels."""
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def add_new_mask(self) -> Tuple[None, str]:
        """Start a new mask id and clear current point selections."""
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self, masks):
        assert len(masks) > 0
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    # -------------------------------------------------------------
    # SAM Model Interaction
    # -------------------------------------------------------------
    def get_sam_features(self) -> Tuple[str, np.ndarray | None]:
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
        self.sam_model.reset_state(self.inference_state)
        msg = (
            "SAM features extracted. Click points to update mask, "
            "and submit when ready to start tracking."
        )
        return msg, self.image

    def set_positive(self) -> str:
        """Set label for next points as positive."""
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking."

    def set_negative(self) -> str:
        """Set label for next points as negative."""
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking."

    def add_point(self, frame_idx: int, i: int, j: int) -> np.ndarray:
        """Add a user-selected point and update corresponding mask."""
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx,
            np.array(self.selected_points, dtype=np.float32),
            np.array(self.selected_labels, dtype=np.int32),
        )
        mask = self.make_index_mask(masks)

        return mask
    

    def get_sam_mask(self, frame_idx: int, input_points: np.ndarray, input_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Get the SAM mask based on the selected points and labels."""
        assert self.sam_model is not None, "SAM model not initialized."

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            _, obj_ids, mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return {
                obj_id: (mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }


    def run_tracker(self) -> Tuple[str, str]:
        """Run the tracker on the video sequence."""
        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
            # index_masks_all.append(self.make_index_mask(masks))

        self.index_masks_all = [self.make_index_mask(v) for k, v in video_segments.items()]

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        msg = f"Wrote current tracked video to {out_vidpath}."
        return out_vidpath, f"{msg} Save the masks to an output directory if it looks good!"

    def save_masks_to_dir(self, output_dir: str) -> str:
        """Save color masks and index masks to a directory."""
        assert self.color_masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask, idx_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = os.path.join(output_dir, name)
            iio.imwrite(out_path, clr_mask)

            base = Path(name).stem
            np_out_path = os.path.join(output_dir, f"{base}.npy")
            np.save(np_out_path, idx_mask)
        
        message = f"Saved masks to {output_dir}!"
        logger.info(message)
        return message


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def is_image(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in (".png", ".jpg", ".jpeg")

def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out

def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    Generate a color palette in HLS space.

    Args:
        n_colors (int): Number of colors to generate
        lightness (float): Lightness value for the HLS color space (0-1). Default is 0.5
        saturation (float): Saturation value for the HLS color space (0-1). Default is 0.7
    
    Returns:
        np.ndarray: Array of space (n_colors, 3) containing RGB colors (uint8).
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h, lightness, saturation) for h in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")

def colorize_masks(images: list[np.ndarray], index_masks: list[np.ndarray], fac: float = 0.5):
    """
    Apply color masks to a list of images based on indexed segmentation masks.

    Args:
        images (list[np.ndarray]): List of original image arrays (H X W X C).
        index_masks (list[np.ndarray]): List of integer index masks (H X W X C).
        flac (float): Blending factor for original image vs. color mask. Default is 0.5

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - out_frames: List of blended image arrays (uint8).
            - color_masks: List of RGB color mask arrays (uint8).
    """
    max_idx = max([m.max() for m in index_masks])
    logger.info(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)

    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks

def compose_img_mask(img: np.ndarray, color_mask: np.ndarray, fac: float = 0.5) -> np.ndarray:
    """
    Blend an image with a color mask using a given factor.

    Args:
        img (np.ndarray): Original image array (H X W X C)
        color_mask (np.ndarray): Color mask array (H X W X C)
        fac (float): Blending factor for the original image. Default 0.5
    
    Returns:
        np.ndarray: Blended image array, dtype uint8
    """
    # Normalize to 0-1 blend, then convert back to uint8
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u

def listdir(path: str) -> List[str]:
    return sorted(os.listdir(path)) if path and os.path.isdir(path) else []

def update_mask_dir(root_dir: str, mask_dir: str, seq_name: str) -> str:
    return os.path.join(root_dir, mask_dir, seq_name)

def select_video_file(root_dir: str, vid_dir: str, video_file: str) -> Tuple[str, str]:
    seq_name = os.path.splitext(video_file)[0]
    logger.info(f"Selected video: {video_file=}")
    video_path = os.path.join(root_dir, vid_dir, video_file)
    return seq_name, video_path

START_INSTRUCTIONS = (
    "Select a video file to extract frames from, "
    "or select an image directory with frames already extracted."
)

def make_demo(
    checkpoint_dir: str,
    model_cfg: str,
    device: str,
    root_dir: str,
    vid_dir_name: str = "videos",
    img_dir_name: str = "images",
    mask_name: str = "masks",
) -> gr.Blocks:
    prompts = PromptGUI(checkpoint_dir, model_cfg, device)
    vid_root = os.path.join(root_dir, vid_dir_name)
    img_root = os.path.join(root_dir, img_dir_name)

    with gr.Blocks() as demo:
        instruction = gr.Textbox(START_INSTRUCTIONS, label="Instruction", interactive=False)
        with gr.Row():
            root_dir_field = gr.Textbox(root_dir, label="Dataset root directory", interactive=False)
            vid_dir_field = gr.Textbox(vid_dir_name, label="Video subdirectory name", interactive=False)
            img_dir_field = gr.Textbox(img_dir_name, label="Image subdirectory name", interactive=False)
            mask_dir_field = gr.Textbox(mask_name, label="Mask subdirectory name", interactive=False)
            seq_name_field = gr.Textbox("", label="Sequence name", interactive=False)

        with gr.Row():
            with gr.Column():
                vid_files = listdir(vid_root)
                vid_files_field = gr.Dropdown(label="Video files", choices=vid_files, value=None)
                input_video_field = gr.Video(label="Input Video", autoplay=False, value=None, height=360)

                with gr.Row():
                    start_time = gr.Number(0, label="Start time (s)")
                    end_time = gr.Number(1, label="End time (s)")
                    sel_fps = gr.Number(30, label="FPS")
                    sel_height = gr.Number(540, label="Height")
                    extract_button = gr.Button("Extract frames")

            with gr.Column():
                img_dirs = listdir(img_root)
                img_dirs_field = gr.Dropdown(label="Image directories", choices=img_dirs, interactive=True, value=None)
                input_img_dir_field = gr.Textbox("", label="Input directory", interactive=False)

                frame_index_slider = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=0, # max is updated later
                    value=0,
                    step=1,
                )
                sam_button = gr.Button("Get SAM features")
                reset_button = gr.Button("Reset")
                current_frame_image = gr.Image(label="Current Frame", value=None)
                with gr.Row():
                    pos_button = gr.Button("Toggle positive")
                    neg_button = gr.Button("Toggle negative")
                clear_button = gr.Button("Clear points")

            with gr.Column():
                output_img = gr.Image(label="Current selection")
                add_mask_button = gr.Button("Add New Mask")
                submit_mask_button = gr.Button("Submit Mask for Tracking")
                final_video_field = gr.Video(label="Masked Video", height=360)
                out_mask_path_field = gr.Textbox(None, label="Path to save masks", interactive=False)
                save_button = gr.Button("Save masks")


        def extract_frames_from_video(
            root_dir: str, vid_dir: str, img_dir: str, vid_file: str, start: float, end: float, fps: int, height: int, ext: str = "jpeg"
        ):
            # SAM2 supports the jpeg folder only
            seq_name = os.path.splitext(vid_file)[0]
            vid_path = os.path.join(root_dir, vid_dir, vid_file)
            out_dir = os.path.join(root_dir, img_dir, seq_name)
            logger.info(f"Extracting frames to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            start_time_str = str(timedelta(seconds=start))
            end_time_str = str(timedelta(seconds=end))
            cmd = (
                    f"ffmpeg -ss {start_time_str} -to {end_time_str} -i {vid_path} "
                f"-vf 'scale=-1:{int(height)},fps={int(fps)}' {out_dir}/%05d.{ext}"
            )
            logger.info(f"Running: {cmd}")
            subprocess.call(cmd, shell=True)
            img_root = os.path.join(root_dir, img_dir)
            img_dirs_updated = listdir(img_root)
            logger.info("Img dirs updated: %s", img_dirs_updated)
            # Update the img dir
            num_imgs = prompts.set_img_dir(out_dir)
            first_img_frame = prompts.set_input_image(0)
            slider_update = gr.update(minimum=0, maximum=max(num_imgs - 1, 0), value=0)
            mask_path = update_mask_dir(root_dir, mask_name, seq_name)
            msg = "Click 'Get SAM Features' and choose the frame you want to annotate."
            return out_dir, gr.update(choices=img_dirs_updated, value=seq_name), slider_update, first_img_frame, mask_path, msg

        def select_image_dir(root_dir: str, img_dir: str, seq_name: str):
            img_dir_path = os.path.join(root_dir, img_dir, seq_name)
            num_imgs = prompts.set_img_dir(img_dir_path)
            slider_update = gr.update(minimum=0, maximum=max(num_imgs - 1, 0), value=0)
            logger.info(f"Selected image dir: {img_dir}")
            input_image_update = prompts.set_input_image(0)
            message = f"Loaded {num_imgs} images from {img_dir_path}."
            return slider_update, input_image_update, message

        def on_frame_select(frame_idx: int) -> Optional[np.ndarray]:
            img = prompts.set_input_image(frame_idx)
            return img

        def on_click_image(evt: gr.SelectData, frame_idx, img):
            if evt.index is None or len(evt.index) < 2:
                logger.info("No valid coordinates selected")
                return img
            j, i = evt.index  # Gradio provides [x, y]
            logger.info(f"Selected coordinates: ({i}, {j}) on frame {frame_idx}")
            index_mask = prompts.add_point(frame_idx, i, j)
            logger.info(f"index_mask shape: {index_mask.shape}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, color_mask)
            out_img = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out_img

        # selecting a video file
        vid_files_field.select(
            select_video_file,
            inputs=[root_dir_field, vid_dir_field, vid_files_field],
            outputs=[seq_name_field, input_video_field],
        )

        # selecting an image directory
        img_dirs_field.select(
            select_image_dir,
            inputs=[root_dir_field, img_dir_field, img_dirs_field],
            outputs=[frame_index_slider, current_frame_image, input_img_dir_field],
        )

        # extracting frames from video
        extract_button.click(
            extract_frames_from_video,
            inputs=[
                root_dir_field,
                vid_dir_field,
                img_dir_field,
                vid_files_field,
                start_time,
                end_time,
                sel_fps,
                sel_height,
            ],
            outputs=[input_img_dir_field, img_dirs_field, frame_index_slider, current_frame_image, out_mask_path_field, instruction],
        )

        frame_index_slider.change(on_frame_select, inputs=[frame_index_slider], outputs=[current_frame_image])
        current_frame_image.select(on_click_image, inputs=[frame_index_slider, current_frame_image], outputs=[output_img])

        sam_button.click(prompts.get_sam_features, outputs=[instruction, current_frame_image])
        reset_button.click(prompts.reset, outputs=[])
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])
        clear_button.click(prompts.clear_points, outputs=[output_img, final_video_field, instruction])
        
        add_mask_button.click(prompts.add_new_mask, outputs=[output_img, instruction])
        submit_mask_button.click(prompts.run_tracker, outputs=[final_video_field, instruction])

        save_button.click(prompts.save_masks_to_dir, inputs=[out_mask_path_field], outputs=[instruction])

    return demo


# Default configuration values
DEFAULTS = {
    "checkpoint_dir": "checkpoints/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "root_dir": "data",
    "vid_dir_name": "videos",
    "img_dir_name": "images",
    "mask_name": "masks",
    "device": "cuda",
    "port": 8890,
}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the SAM demo with the specified configuration.")
    parser.add_argument("--port", type=int, default=DEFAULTS["port"], help="Port to run the demo server.")
    parser.add_argument("--device", type=str, default=DEFAULTS["device"], choices=["cuda", "cpu"], help="Device to use for processing.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULTS["checkpoint_dir"], help="Path to the model checkpoint.")
    parser.add_argument("--model_cfg", type=str, default=DEFAULTS["model_cfg"], help="Path to the model configuration file.")
    parser.add_argument("--root_dir", type=str, default=DEFAULTS["root_dir"], help="Root directory for the demo.")
    parser.add_argument("--vid_dir_name", type=str, default=DEFAULTS["vid_dir_name"], help="Subdirectory name for videos.")
    parser.add_argument("--img_dir_name", type=str, default=DEFAULTS["img_dir_name"], help="Subdirectory name for images.")
    parser.add_argument("--mask_name", type=str, default=DEFAULTS["mask_name"], help="Name for the output masks.")
    
    return parser.parse_args()

def check_device(args):
    """Check and adjust the device if CUDA is not available."""
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Switching to CPU.")
        args.device = "cpu"
    return args

if __name__ == "__main__":
    args = parse_args()
    args = check_device(args)

    demo = make_demo(
        args.checkpoint_dir,
        args.model_cfg,
        args.device.lower(),
        args.root_dir,
        args.vid_dir_name,
        args.img_dir_name,
        args.mask_name,
    )

    demo.launch(server_port=args.port, debug=True)
