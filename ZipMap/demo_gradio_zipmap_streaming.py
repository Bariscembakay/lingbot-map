# Code for ZipMap (CVPR 2026); created by Haian Jin

model_config = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": False,
    "enable_camera_mlp": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": 
        {
            "ttt_mode": True,
            "params": {
                "bias": True,   
                "head_dim": 1024,
                "inter_multi": 2,
                "base_lr": 0.01,
                "muon_update_steps": 5,
                "use_gate_fn": True
            },
            "window_size": 1
        },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True, 
    }
}
BASE_CACHE_DIR = "./debug_testing/demo_cache/"

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import argparse 
sys.path.append("zipmap/")

from visual_util import predictions_to_glb
from zipmap.models.ZipMap_AR import ZipMap
from zipmap.utils.load_fn import load_and_preprocess_images
from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
from zipmap.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3, homogenize_points

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading ZipMap model...")

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default="./checkpoints/checkpoint_online.pt", help='Path to the model checkpoint')
parser.add_argument('--ema', action='store_true', help='Use EMA weights if available')
parser.add_argument('--align_first_view', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help='Align output point cloud to the first view coordinate system (default: True)')
args = parser.parse_args()

model = ZipMap(**model_config)
ckpt_path = args.ckpt_path
checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=True)
if args.ema and "ema" in checkpoint:
    print("Using EMA weights from checkpoint.")
    model_state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint
else:
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

missing, unexpected = model.load_state_dict(model_state_dict, strict=False)

if missing:
    print(f"Missing keys in state_dict: {missing}")
if unexpected:
    print(f"Unexpected keys in state_dict: {unexpected}")

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the ZipMap model on images in the 'target_dir/images' folder and return predictions.
    Args:
        target_dir: Directory containing the images subfolder
        model: The ZipMap model
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list
    predictions['pose_enc_mlp_list'] = None

    # Align all cameras (and subsequent world points) to the first view's coordinate frame
    if args.align_first_view:
        print("Aligning to first view coordinate frame...")
        S_frames = predictions["extrinsic"].shape[0]
        extrinsics_homog = np.concatenate(
            [predictions["extrinsic"], np.zeros((S_frames, 1, 4), dtype=predictions["extrinsic"].dtype)],
            axis=-2,
        )  # (S, 4, 4)
        extrinsics_homog[:, -1, -1] = 1.0
        first_cam_inv = closed_form_inverse_se3(extrinsics_homog[0:1])[0]  # (4, 4)
        new_extrinsics_homog = extrinsics_homog @ first_cam_inv  # (S, 4, 4)
        predictions["extrinsic"] = new_extrinsics_homog[:, :3, :]  # (S, 3, 4)
    else:
        print("Not aligning to first view coordinate frame. Output will be in the original camera coordinate system predicted by the model.")

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)

    world_points_from_depth = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points_from_depth

    world_points_from_local_points = None
    if "local_points" in predictions:
        local_points = predictions["local_points"]  # (S, H, W, 3)
        local_points_conf = predictions["depth_conf"]  # (S, H, W)
        cam_to_world_extrinsic = closed_form_inverse_se3(predictions["extrinsic"]) # (S, 4, 4)
        world_points_from_local_points = np.einsum('sij, shwj -> shwi', cam_to_world_extrinsic, homogenize_points(local_points))
        predictions["world_points"] = world_points_from_local_points[..., :3]
        predictions["world_points_conf"] = local_points_conf

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, target_fps=None):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).

    Args:
        input_video: Uploaded video file
        input_images: Uploaded image files
        target_fps: Target frames per second for video sampling (default: 1.0)
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(BASE_CACHE_DIR, f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        video_fps = vs.get(cv2.CAP_PROP_FPS)

        # Use target_fps if specified, otherwise default to 1 FPS
        if target_fps is not None and target_fps > 0:
            fps_target = float(target_fps)
        else:
            fps_target = 1.0  # Default: 1 frame per second

        # Calculate frame interval based on target FPS
        # frame_interval = video_fps / target_fps
        # For example: 30fps video, target 2fps -> sample every 15 frames
        frame_interval = max(1, int(round(video_fps / fps_target)))

        print(f"Video FPS: {video_fps}, Target FPS: {fps_target}, Frame interval: {frame_interval}")

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, target_fps):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, target_fps)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    cam_size_factor=1.0,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_camfactor{cam_size_factor}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        cam_size_factor=float(cam_size_factor),
    )

    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example,
    cam_size_factor=1.0,
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_camfactor{cam_size_factor}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
            cam_size_factor=float(cam_size_factor),
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------


dl3dv_video = "examples/videos/dl3dv.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
pyramid_video = "examples/videos/pyramid.mp4"
drift_straight_video = "examples/videos/drift-straight.mp4"
sora_big_sur_video = "examples/videos/sora_big-sur.mp4"
Istanbul_video = "examples/videos/Istanbul.mp4"
figureskating_video = "examples/videos/figureskating.mp4"
walkthrough_video = "examples/videos/walkthrough.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
    """
    <h1>🤐 ZipMap Playground</h1>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. Our model takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. For videos, you can adjust the "Video Sampling FPS" to control how many frames per second are extracted (default: 1 FPS).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Camera Frustum Size:</em> Scale the size of the camera frustum visualizations.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background / Filter White Background:</em> Remove sky, black-background, or white-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Local Pointmap and Camera Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">Our model typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of our model's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            target_fps = gr.Number(
                label="Video Sampling FPS",
                value=1.0,
                minimum=0.1,
                maximum=30,
                step=0.5,
                info="Target frames per second for video sampling (default: 1.0 FPS) \n You need to re-upload the video after changing this value."
            )

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5, camera_position=(-90, 90, 3.0))

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Local Pointmap and Camera Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                cam_size_factor = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Camera Frustum Size")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [Istanbul_video, "44", None, 3.0, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],
        # [sora_big_sur_video, "19", None, 3.0, 30.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],
        [pyramid_video, "30", None, 1.0, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],
        [room_video, "8", None, 1.0, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],
        [dl3dv_video, "29", None, 2.0, 30.0, False, False, True, False, "Depthmap and Camera Branch", "True", 0.5],
        [kitchen_video, "29", None, 2.0, 40.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],
        [figureskating_video, "14", None, 3.0, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True", 0.5],
        [single_cartoon_video, "1", None, 1.0, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True", 1.0],

    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        target_fps,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
        cam_size_factor=1.0,

    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_video, input_images, target_fps)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, cam_size_factor
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            target_fps,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
            cam_size_factor,

        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            cam_size_factor,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    viz_inputs = [
        target_dir_output,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example,
        cam_size_factor,
    ]
    viz_outputs = [reconstruction_output, log_output]

    conf_thres.change(update_visualization, viz_inputs, viz_outputs)
    frame_filter.change(update_visualization, viz_inputs, viz_outputs)
    mask_black_bg.change(update_visualization, viz_inputs, viz_outputs)
    mask_white_bg.change(update_visualization, viz_inputs, viz_outputs)
    show_cam.change(update_visualization, viz_inputs, viz_outputs)
    mask_sky.change(update_visualization, viz_inputs, viz_outputs)
    prediction_mode.change(update_visualization, viz_inputs, viz_outputs)
    cam_size_factor.change(update_visualization, viz_inputs, viz_outputs)

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, target_fps],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, target_fps],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    target_fps.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, target_fps],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
