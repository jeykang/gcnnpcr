# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import tempfile

import omni
import omni.kit.commands
import omni.replicator.core as rep
import torch
import os
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Configure logger
logger = logging.getLogger(__name__)

def compute_normals_pca(points, k=16):
    """Compute normals using PCA on k-nearest neighbors."""
    # Convert to numpy for k-nearest neighbors
    points_np = points.cpu().numpy()
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_np)
    distances, indices = nbrs.kneighbors(points_np)
    
    # Compute normals for each point
    normals = []
    for i in range(len(points_np)):
        neighbors = points_np[indices[i]]
        # Center the neighbors
        centered = neighbors - neighbors.mean(axis=0)
        # Compute covariance matrix
        cov = np.dot(centered.T, centered)
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is eigenvector corresponding to smallest eigenvalue
        normal = eigenvectors[:, 0]
        # Ensure consistent orientation (optional)
        if normal[2] < 0:
            normal = -normal
        normals.append(normal)
    
    return torch.from_numpy(np.array(normals)).float()

###############################################################################
# SCENARIO CLASS
###############################################################################
class LidarCompletionScenario:
    """
    Creates a rotary LiDAR sensor (IsaacSensorCreateRtxLidar) and attaches an annotator
    (RtxSensorCpuIsaacCreateRTXLidarScanBuffer) to fetch the point cloud data.

    On every full rotation (~360 degrees), the scenario:
      1) Fetches the LiDAR data from the annotator.
      2) Passes it to the loaded PyTorch model for inference.
      3) Saves the model output to the user-specified directory.
    """

    def __init__(self):
        self._lidar_prim_path = "/World/RTXLidarSensor"
        self._render_product = None

        # For debug visualization (optional)
        self._debug_draw_writer = None

        # For retrieving LiDAR data
        self._annotator = None

        self._time_accumulator = 0.0   # Track how much time has passed since last inference
        self._inference_period = 10.0  # Seconds between inferences (change as desired)

        # Model application
        self._apply_model = False
        self._model = None
        self._model_path = None
        self._output_dir = None
        self._inference_callback = None  # We'll store a function pointer here

    def set_inference_callback(self, callback_fn):
        """
        Let external code (e.g. the UI) register a function that we call every time inference completes.
        callback_fn should accept a single argument (the path to the .pt file).
        """
        self._inference_callback = callback_fn

    def load_example_assets(self):
        """Optionally load environment or assets. Return references for World to add."""
        return []

    def setup(self):
        """
        Called once after "Load" is pressed in the extension UI.
        Creates the LiDAR sensor, the debug-draw pipeline, and the data annotator.
        """
        self._create_lidar_sensor()
        self.reset()

    def reset(self):
        self._time_accumulator = 0.0

    def update(self, step):
        """
        Called every physics frame while the timeline is playing.
        'step' is the size of the physics step in seconds (e.g., 1/60 if your physics_dt=1/60).
        """
        # 1) Accumulate simulation time
        self._time_accumulator += step

        # 2) If we've hit or exceeded the 10-second mark, trigger inference
        if self._time_accumulator >= self._inference_period:
            self._time_accumulator -= self._inference_period  # or just reset to 0.0
            if self._apply_model and self._model is not None:
                self._run_inference_and_save()

        return False

    def set_model_path(self, model_path):
        self._model_path = model_path

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    def load_model(self):
        # Ensure the file exists
        if not self._model_path or not os.path.exists(self._model_path):
            logger.info(f"[LidarCompletionScenario] Invalid model path: {self._model_path}")
            self._model = None
            return

        try:
            logger.info(f"[LidarCompletionScenario] Loading state dict from {self._model_path} ...")
            
            # 1) Import your model class. Adjust if minimal_main_4.py is in another directory.
            from .minimal_main_4 import FullModelSnowflake
            
            # 2) Instantiate the model with the same constructor args you used in training:
            self._model = FullModelSnowflake(
                g_hidden_dims=[64, 128],
                g_out_dim=128,
                t_d_model=128,
                t_nhead=8,
                t_layers=4,
                coarse_num=64,
                radius=1.0
            )
            
            # 3) Load the state dict
            state_dict = torch.load(self._model_path, map_location="cpu")
            self._model.load_state_dict(state_dict)
            
            # 4) Switch to evaluation mode for inference
            self._model.eval()
            
            logger.info("[LidarCompletionScenario] Model loaded successfully (from state dict).")
        except Exception as e:
            logger.info(f"[LidarCompletionScenario] Error loading model: {e}")
            self._model = None

    def enable_model_application(self, enable: bool):
        """
        If True, we run inference on each full rotation.
        """
        self._apply_model = enable

    ###########################################################################
    # Internal Helpers
    ###########################################################################
    def _create_lidar_sensor(self):
        """
        1) Create a LiDAR sensor with IsaacSensorCreateRtxLidar, config="Example_Rotary".
        2) Create a Hydra texture (render product).
        3) Create a debug-draw pipeline (optional).
        4) Create and initialize the RtxSensorCpuIsaacCreateRTXLidarScanBuffer annotator 
           so we can fetch the point cloud data in `_run_inference_and_save()`.
        """
        lidar_config = "Example_Rotary"
        result, sensor_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=self._lidar_prim_path,
            parent=None,
            config=lidar_config,
            translation=(0, 0, 1.0),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        )
        if not result or sensor_prim is None:
            logger.info("[LidarCompletionScenario] Failed to create RTX Lidar sensor.")
            return

        # (2) Hydra render product
        self._render_product = rep.create.render_product(sensor_prim.GetPath(), [1, 1], name="Isaac")
        if not self._render_product:
            logger.info("[LidarCompletionScenario] Failed to create render product for LiDAR.")
            return

        # (3) Debug-draw pipeline (for visual point cloud in the viewport)
        self._debug_draw_writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        if self._debug_draw_writer:
            self._debug_draw_writer.attach(self._render_product)
        else:
            logger.info("[LidarCompletionScenario] Could not create debug-draw writer. Visualization is disabled.")

        # (4) Create & initialize the annotator so we can retrieve point cloud data
        self._annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        if self._annotator:
            # Configure the annotator. For example, we want world-space positions, 
            # plus distance, intensity, etc. (You can enable or disable fields as needed.)
            self._annotator.initialize(
                transformPoints=True,
                keepOnlyPositiveDistance=True,
                outputDistance=True,
                outputIntensity=True,
                outputAzimuth=True,
                outputElevation=True,
                outputNormal=False,       # or True if you want surface normals
                outputObjectId=False,     # or True if you need the object IDs
                outputTimestamp=False,    # or True if you need time info
            )
            self._annotator.attach(self._render_product)
        else:
            logger.info("[LidarCompletionScenario] Could not create RtxSensorCpuIsaacCreateRTXLidarScanBuffer annotator.")

    def _lidar_exists(self):
        stage = get_current_stage()
        return stage.GetPrimAtPath(self._lidar_prim_path).IsValid()

    def _run_inference_and_save(self):
        """
        1) Get LiDAR data
        2) Preprocess data to match training format
        3) Run inference
        4) Save results
        """
        if not self._annotator:
            logger.info("[LidarCompletionScenario] Annotator not available. No data to fetch.")
            return

        if not self._output_dir or not os.path.exists(self._output_dir):
            logger.info(f"[LidarCompletionScenario] Invalid output directory: {self._output_dir}")
            return

        # Get LiDAR data
        lidar_data = self._annotator.get_data()
        logger.info("[LidarCompletionScenario] Retrieved annotator data")
        if not lidar_data or "data" not in lidar_data:
            logger.info("[LidarCompletionScenario] No 'data' key in annotator output.")
            return

        points_np = lidar_data["data"]  # Nx3 float array
        if points_np is None or len(points_np) == 0:
            logger.info("[LidarCompletionScenario] LiDAR returned 0 points.")
            return

        # Preprocess points to match training format
        N = points_np.shape[0]
        num_points = 8192  # Same as training

        # 1. Downsample to num_points
        if N < num_points:
            indices = np.random.choice(N, num_points, replace=True)
        else:
            indices = np.random.choice(N, num_points, replace=False)
        
        coords = points_np[indices]

        # 2. Normalize coordinates
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c)/2
        scale = (max_c - min_c).max()/2
        if scale < 1e-8:
            scale = 1.0
        coords = (coords - center)/scale

        # 3. Convert to tensor and compute normals
        coords_t = torch.from_numpy(coords).float()
        normals_t = compute_normals_pca(coords_t, k=16)  # Same k as training

        # 4. Combine coordinates and normals
        full_6d = torch.cat([coords_t, normals_t], dim=-1)  # => [N,6]

        try:
            logger.info("[LidarCompletionScenario] Running data through model...")
            # Add batch dimension for model
            with torch.no_grad():
                model_input = full_6d.unsqueeze(0)  # [1,N,6]
                model_output = self._model(model_input).permute(0, 2, 1).contiguous()
        except Exception as e:
            logger.info(f"[LidarCompletionScenario] Model inference error: {e}")
            return

        # Save both input and output
        out_path = os.path.join(self._output_dir, "lidar_model_output.pt")
        torch.save({
            'input': full_6d,
            'output': model_output,
            'center': center,
            'scale': scale
        }, out_path)
        logger.info(f"[LidarCompletionScenario] Saved model input/output to: {out_path}")

        if self._inference_callback:
            self._inference_callback(out_path)

###############################################################################
# EXTENSION UI - SAME TEMPLATE, ADDED ANNOTATOR LOGIC
###############################################################################
class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()

        self._scenario = LidarCompletionScenario()

        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build the UI with:
          - LOAD, RESET
          - Model path + output directory + APPLY MODEL
          - RUN scenario button
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1/60.0, rendering_dt=1/60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        model_frame = CollapsableFrame("Model & Lidar Inference", collapsed=False)
        with model_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(spacing=5):
                    ui.Label("Model Path:", width=100)
                    self._model_path_field = ui.StringField(height=0)
                with ui.HStack(spacing=5):
                    ui.Label("Output Dir:", width=100)
                    self._output_dir_field = ui.StringField(height=0)
                ui.Button("APPLY MODEL", height=0, clicked_fn=self._on_apply_model_clicked)

        run_scenario_frame = CollapsableFrame("Run Scenario")
        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        inference_results_frame = CollapsableFrame("Inference Results", collapsed=False)
        with inference_results_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Label that shows textual info
                self._inference_label = ui.Label("No inference results yet.", height=0)
                
                # This is where we'll display a Matplotlib plot
                # We'll later update self._inference_image.source_url 
                # each time inference completes
                self._inference_image = ui.Image(width=256, height=256, fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)

    ######################################################################################
    # Extension Lifecycle
    ######################################################################################
    def _on_init(self):
        pass

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        SingleXFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        create_new_stage()
        self._add_light_to_stage()

        loaded_objects = self._scenario.load_example_assets()
        world = World.instance()
        for obj in loaded_objects:
            world.scene.add(obj)

    def _setup_scenario(self):
        self._scenario.setup()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

        # Register a callback so that every time the scenario finishes inference,
        # it calls our _on_inference_complete function
        self._scenario.set_inference_callback(self._on_inference_complete)

    def _on_post_reset_btn(self):
        self._scenario.reset()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._scenario = LidarCompletionScenario()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    def _on_apply_model_clicked(self):
        model_path = self._model_path_field.model.get_value_as_string()
        output_dir = self._output_dir_field.model.get_value_as_string()

        self._scenario.set_model_path(model_path)
        self._scenario.set_output_dir(output_dir)
        self._scenario.load_model()

        self._scenario.enable_model_application(True)

        # Optionally auto-play
        if not self._timeline.is_playing():
            self._timeline.play()

    def _on_inference_complete(self, pt_path):
        """
        Called each time the scenario saves a .pt file. We'll:
        1) Load the data,
        2) Create a single figure with two 3D scatter plots side-by-side: input vs output,
        3) Save the figure to a temp .png,
        4) Display that image in our extension UI.
        """
        if not os.path.exists(pt_path):
            self._inference_label.text = f"File {pt_path} not found!"
            return
        
        try:
            # 1) Load the saved data (input, output, center, scale, etc.)
            data = torch.load(pt_path, map_location="cpu")
            input_tensor = data.get("input", None)   # shape [N,6] (xyz + normal)
            output_tensor = data.get("output", None) # shape could vary, e.g. [N,3] or [1,N,3]

            if input_tensor is None:
                self._inference_label.text = "No 'input' found in PT file."
                return

            # -- Extract input points (x,y,z) from the first 3 channels of 'input'
            points_in = input_tensor.detach().cpu()[:, :3]  # [N,3]
            logger.info("[LidarCompletionScenario] Input tensor shape: %s", points_in.shape)
            x_in = points_in[:, 0].numpy()
            y_in = points_in[:, 1].numpy()
            z_in = points_in[:, 2].numpy()

            # -- Extract output points
            if output_tensor is not None:
                # If your model_output is shape [1,N,3], remove the batch dim:
                if output_tensor.dim() == 3 and output_tensor.size(0) == 1:
                    output_tensor = output_tensor.squeeze(0)  # now [N,3]
                points_out = output_tensor.detach().cpu()[..., :3]  # if it's Nx3
                logger.info("[LidarCompletionScenario] Output tensor shape: %s", points_out.shape)
                x_out = points_out[:, 0].detach().cpu().numpy()
                y_out = points_out[:, 1].detach().cpu().numpy()
                z_out = points_out[:, 2].detach().cpu().numpy()
            else:
                x_out = y_out = z_out = None

            # 2) Create a single figure with two subplots, side by side
            fig = plt.figure(figsize=(8,4), dpi=100)

            # Left subplot for Input
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(x_in, y_in, z_in, s=1, c='blue')
            ax1.set_title("Input")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.view_init(elev=30, azim=-60)  # Optional: tweak 3D view angle

            # Right subplot for Output (if available)
            ax2 = fig.add_subplot(122, projection='3d')
            if x_out is not None:
                ax2.scatter(x_out, y_out, z_out, s=1, c='red')
            ax2.set_title("Output" if x_out is not None else "No Output")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            ax2.view_init(elev=30, azim=-60)

            # Optionally unify coordinate scales so both subplots match
            # Combine all points
            if x_out is not None:
                combined = torch.cat([points_in, points_out], dim=0).numpy()
            else:
                combined = points_in.numpy()
            min_vals = combined.min(axis=0)
            max_vals = combined.max(axis=0)
            for ax in [ax1, ax2]:
                ax.set_xlim(min_vals[0], max_vals[0])
                ax.set_ylim(min_vals[1], max_vals[1])
                ax.set_zlim(min_vals[2], max_vals[2])

            # 3) Save the figure to a temporary .png file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                temp_path = tmp_file.name
            plt.savefig(temp_path)
            plt.close(fig)

            # 4) Update the UI label and image
            self._inference_label.text = f"Side-by-side 3D plot loaded from {pt_path}"
            self._inference_image.source_url = temp_path

        except Exception as e:
            self._inference_label.text = f"Error loading {pt_path}: {e}"
