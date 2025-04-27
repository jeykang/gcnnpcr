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
from isaacsim.core.prims import XFormPrim # Use XFormPrim for general transformations
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, Gf, UsdGeom, Vt, UsdShade # Import UsdShade for materials
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
from scipy.spatial import ConvexHull # Import ConvexHull
from scipy.spatial.distance import cdist # For distance calculation

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
        self._show_completed_mesh = True # Controls visibility of the output mesh overlay

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
        self._remove_previous_visualization() # Clean up old visualizations on reset

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
            # Make sure the model definition matches the saved state_dict
            try:
                # Try importing the specific model used during training
                from .minimal_main_4 import FullModelSnowflake
            except ImportError:
                logger.error("Could not import FullModelSnowflake from minimal_main_4.py. Ensure the file exists and is correct.")
                self._model = None
                return

            # 2) Instantiate the model with the same constructor args used during training
            #    These parameters MUST match the saved model's architecture.
            #    If you change the model, update these parameters accordingly.
            self._model = FullModelSnowflake(
                g_hidden_dims=[64, 128],
                g_out_dim=128,
                t_d_model=128,
                t_nhead=8,
                t_layers=4,
                coarse_num=64, # Example value, adjust if needed
                radius=1.0     # Example value, adjust if needed
                # Add other parameters if your FullModelSnowflake requires them
            )

            # 3) Load the state dict
            state_dict = torch.load(self._model_path, map_location="cpu")
            self._model.load_state_dict(state_dict)

            # 4) Switch to evaluation mode for inference
            self._model.eval()

            logger.info("[LidarCompletionScenario] Model loaded successfully (from state dict).")
        except Exception as e:
            logger.error(f"[LidarCompletionScenario] Error loading model: {e}", exc_info=True)
            self._model = None

    def enable_model_application(self, enable: bool):
        """
        If True, we run inference on each full rotation.
        """
        self._apply_model = enable

    def _remove_previous_visualization(self):
        """Removes previously generated mesh visualization"""
        stage = get_current_stage()
        mesh_path = "/World/ReconstructedMesh"
        if stage.GetPrimAtPath(mesh_path).IsValid():
            omni.kit.commands.execute("DeletePrims", paths=[mesh_path], destructive=True)

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

    def _create_mesh_visualization(self, points_np):
        """Creates a UsdGeom.Mesh prim using Convex Hull and colors vertices based on nearest scene mesh."""
        if len(points_np) < 4: # Need at least 4 points for 3D convex hull
            logger.info("[LidarCompletionScenario] Not enough points for mesh visualization.")
            return
        try:
            # --- Convex Hull remains the same ---
            hull = ConvexHull(points_np)
            # Use all points from the convex hull as vertices for color calculation
            vertices_world = points_np[hull.vertices]
            faces = hull.simplices # Indices into vertices_world

            stage = get_current_stage()
            mesh_path = "/World/ReconstructedMesh"
            mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)

            # --- Set basic mesh properties ---
            points_vt = Vt.Vec3fArray.FromNumpy(vertices_world.astype(np.float32))
            mesh_prim.CreatePointsAttr().Set(points_vt)

            face_vertex_counts = Vt.IntArray([3] * len(faces)) # All triangles
            mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)

            face_vertex_indices = Vt.IntArray(faces.flatten().astype(np.int32))
            mesh_prim.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)

            # --- Vertex Color Calculation ---
            target_prims = []
            target_centers_world = []
            target_colors = []
            default_color = Gf.Vec3f(0.5, 0.5, 0.5) # Default gray if no color found

            # Iterate through all prims to find potential target meshes
            for prim in stage.Traverse():
                if prim.GetPath() == mesh_path: # Skip self
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    try:
                        # Get world bounding box
                        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
                        prim_bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
                        center_world = prim_bbox.GetMidpoint()

                        # Get displayColor (or default)
                        color_attr = prim.GetDisplayColorAttr()
                        color = color_attr.Get()[0] if color_attr and color_attr.HasValue() and color_attr.Get() else default_color
                        # TODO: Could also try getting material color if displayColor is not set

                        target_prims.append(prim)
                        target_centers_world.append(np.array(center_world))
                        target_colors.append(color)
                    except Exception as e:
                        logger.warning(f"Could not process target prim {prim.GetPath()}: {e}")

            vertex_colors_list = []
            if target_centers_world:
                target_centers_np = np.array(target_centers_world) # Shape [M, 3]
                # Calculate distances between all vertices and all target centers
                # vertices_world shape: [N, 3], target_centers_np shape: [M, 3]
                dist_matrix = cdist(vertices_world, target_centers_np) # Shape [N, M]
                # Find the index of the closest target center for each vertex
                nearest_indices = np.argmin(dist_matrix, axis=1) # Shape [N]

                # Assign colors based on the nearest prim
                for idx in nearest_indices:
                    vertex_colors_list.append(target_colors[idx])
            else:
                # If no target meshes found, assign default color to all vertices
                logger.info("[LidarCompletionScenario] No target meshes found for color coding. Using default color.")
                vertex_colors_list = [default_color] * len(vertices_world)

            # --- Apply Vertex Colors ---
            if vertex_colors_list:
                vertex_colors_vt = Vt.Vec3fArray(vertex_colors_list)
                # Create or get the primvar for displayColor
                color_primvar = mesh_prim.CreatePrimvar("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
                color_primvar.Set(vertex_colors_vt)
                # color_primvar.SetInterpolation(UsdGeom.Tokens.vertex) # Set interpolation explicitly if needed, often inferred

            # Remove the single mesh color attribute if it exists or was previously set
            # mesh_prim.RemoveProperty("primvars:displayColor") # If previously set with different interpolation
            if mesh_prim.GetAttribute("displayColor").IsValid():
                 mesh_prim.GetAttribute("displayColor").Clear() # Clear single color if set

            # Optional: Make it slightly transparent (requires material setup usually)
            # primvarsApi = UsdGeom.PrimvarsAPI(mesh_prim)
            # primvarsApi.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set([0.8] * len(vertices_world))


            logger.info(f"[LidarCompletionScenario] Created mesh visualization at {mesh_path} with vertex colors.")
        except Exception as e:
            # Convex hull can fail if points are degenerate (e.g., co-planar)
            logger.error(f"[LidarCompletionScenario] Error creating mesh visualization: {e}", exc_info=True)

    def _toggle_completed_mesh_visibility(self, show):
        """Toggle the visibility of the reconstructed mesh overlay"""
        self._show_completed_mesh = show
        stage = get_current_stage()
        mesh_path = "/World/ReconstructedMesh"

        # Toggle mesh visibility
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if mesh_prim.IsValid():
            imageable = UsdGeom.Imageable(mesh_prim)
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if show else UsdGeom.Tokens.invisible)

    def _lidar_exists(self):
        stage = get_current_stage()
        return stage.GetPrimAtPath(self._lidar_prim_path).IsValid()

    def _run_inference_and_save(self):
        """
        1) Get LiDAR data
        2) Preprocess data to match training format
        3) Run inference
        4) Save results
        5) Visualize results in the scene
        """
        # Steps 1 & 2: Get and preprocess data (moved to a separate function for clarity)
        preprocessed_data = self._get_and_preprocess_lidar_data()
        if preprocessed_data is None:
            return
        full_6d, center, scale = preprocessed_data

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

        # 5. Visualize the output in the scene
        self._visualize_output_in_scene(model_output, center, scale)

    def _get_and_preprocess_lidar_data(self):
        """Gets LiDAR data, preprocesses it, and returns tensors and normalization info."""
        if not self._annotator:
            logger.info("[LidarCompletionScenario] Annotator not available. No data to fetch.")
            return None

        # Get LiDAR data
        lidar_data = self._annotator.get_data()
        logger.info("[LidarCompletionScenario] Retrieved annotator data")
        if not lidar_data or "data" not in lidar_data:
            logger.info("[LidarCompletionScenario] No 'data' key in annotator output.")
            return None

        points_np = lidar_data["data"]  # Nx3 float array
        if points_np is None or len(points_np) == 0:
            logger.info("[LidarCompletionScenario] LiDAR returned 0 points.")
            return None

        # Preprocess points to match training format
        N = points_np.shape[0]
        num_points = 4096 # Match training num_points

        # 1. Downsample/Upsample to num_points
        if N == 0:
            logger.warning("[LidarCompletionScenario] No points received from LiDAR.")
            return None
        elif N < num_points:
            indices = np.random.choice(N, num_points, replace=True)
        else:
            indices = np.random.choice(N, num_points, replace=False)
        coords = points_np[indices]

        # 2. Normalize coordinates
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        scale = np.max(np.linalg.norm(coords_centered, axis=1))
        if scale < 1e-8: scale = 1.0 # Avoid division by zero
        coords_normalized = coords_centered / scale

        # 3. Convert to tensor and compute normals (on normalized data)
        coords_t = torch.from_numpy(coords_normalized).float()
        normals_t = compute_normals_pca(coords_t, k=16)  # Same k as training

        # 4. Combine coordinates and normals
        full_6d = torch.cat([coords_t, normals_t], dim=-1)  # => [N,6]

        return full_6d, center, scale # Return normalization info

    def _visualize_output_in_scene(self, model_output, center, scale):
        """
        De-normalizes the model output and adds it to the scene as a mesh overlay.
        Args:
            model_output: The raw output tensor from the model. [B, C, N] or [B, N, C]
            center: Original center used for normalization
            scale: Original scale used for normalization
        """
        if model_output is None:
            logger.info("[LidarCompletionScenario] No model output to visualize")
            return

        # 1. De-normalize points back to world space
        # Handle potential batch dimension and channel dimension
        if model_output.dim() == 3:
            if model_output.size(1) == 3: # Shape [B, 3, N]
                output_points = model_output.squeeze(0).permute(1, 0) # -> [N, 3]
            elif model_output.size(2) == 3: # Shape [B, N, 3]
                output_points = model_output.squeeze(0) # -> [N, 3]
            else:
                 logger.error(f"Unexpected model output shape: {model_output.shape}")
                 return
        elif model_output.dim() == 2 and model_output.size(1) == 3: # Shape [N, 3]
            output_points = model_output
        else:
            logger.error(f"Unexpected model output shape: {model_output.shape}")
            return

        points_np = output_points.detach().cpu().numpy()

        # Apply inverse normalization
        points_np = points_np * scale + center

        # 2. Remove previous visualizations
        self._remove_previous_visualization()

        # 3. Create new mesh visualization if enabled
        if self._show_completed_mesh:
            self._create_mesh_visualization(points_np)

        logger.info(f"[LidarCompletionScenario] Visualized output mesh with {points_np.shape[0]} points.")

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
        """Callback for Timeline events (Play, Pause, Stop)."""
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # Ensure visualization is removed when simulation stops
            if self._scenario:
                self._scenario._remove_previous_visualization()
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
             # If paused, keep visualization but stop updates
             pass
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
             # Ensure button state is correct if played from outside the extension
             if hasattr(self, "_scenario_state_btn"):
                 self._scenario_state_btn.set_state_b() # Set to STOP

    def on_physics_step(self, step: float):
        """Forward the physics step to the scenario's update method."""
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        """Clean up resources."""
        # Remove visualization prims
        if self._scenario:
            self._scenario._remove_previous_visualization()
        # Clean up UI elements
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        self.wrapped_ui_elements.clear()
        # Reset internal state if necessary
        self._scenario = None # Or re-initialize if needed

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

                # Add a toggle for visualization
                with ui.HStack(spacing=5):
                    ui.Label("Show Completed Mesh:", width=130) # Updated Label
                    self._show_completed_mesh_checkbox = ui.CheckBox( # Renamed variable
                        model=ui.SimpleBoolModel(self._scenario._show_completed_mesh) # Use scenario's state
                    )
                    # Link checkbox changes to the scenario's visibility toggle method
                    self._show_completed_mesh_checkbox.model.add_value_changed_fn(self._on_show_completed_mesh_changed) # Renamed callback

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
        self._timeline = omni.timeline.get_timeline_interface()
        # Keep a reference to the scenario instance
        self._scenario = LidarCompletionScenario()
        # Pass the UI update function to the scenario
        self._scenario.set_inference_callback(self._on_inference_complete)

    def _add_light_to_stage(self):
        """Adds a default light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        # Make the light visible in the stage tree
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
        # Reset the scenario state after loading
        self._scenario.reset()

    def _setup_scenario(self):
        """Called after scene is loaded or reset, prepares the scenario."""
        if not self._scenario:
             self._scenario = LidarCompletionScenario()
             self._scenario.set_inference_callback(self._on_inference_complete)

        self._scenario.setup()
        # Update UI state

        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

        # Register a callback so that every time the scenario finishes inference,
        # it calls our _on_inference_complete function
        self._scenario.set_inference_callback(self._on_inference_complete)

    def _on_post_reset_btn(self):
        """Callback when reset button is clicked."""
        # Reset the scenario
        self._scenario.reset()
        # Reset the UI button state
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """Callback for physics steps, forwards to scenario."""
        if not self._scenario:
            return
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        """Resets the entire extension state, often called on stage change."""
        # Clean up existing scenario and visualizations
        if self._scenario:
            self._scenario._remove_previous_visualization()
            # Potentially add more scenario cleanup if needed

        # Re-initialize the scenario
        self._scenario = LidarCompletionScenario()
        self._scenario.set_inference_callback(self._on_inference_complete)

        # Reset UI elements
        self._reset_ui()

    def _reset_ui(self):
        """Resets the UI elements to their initial state."""
        if hasattr(self, "_scenario_state_btn"):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        if hasattr(self, "_reset_btn"):
            self._reset_btn.enabled = False
        self._inference_label.text = "No inference results yet."
        self._inference_image.source_url = "" # Clear the image

    def _on_apply_model_clicked(self):
        model_path = self._model_path_field.model.get_value_as_string()
        output_dir = self._output_dir_field.model.get_value_as_string()

        self._scenario.set_model_path(model_path)
        self._scenario.set_output_dir(output_dir)
        self._scenario.load_model()

        self._scenario.enable_model_application(True)
        logger.info(f"Model applied: {model_path}, Output dir: {output_dir}")

        # Optionally auto-play
        if not self._timeline.is_playing():
            self._timeline.play()
            # Ensure the run button state reflects playing state
            if hasattr(self, "_scenario_state_btn"):
                 self._scenario_state_btn.set_state_b() # Set to STOP state
                 self._scenario_state_btn.enabled = True

    def _on_show_completed_mesh_changed(self, model):
        self._scenario._toggle_completed_mesh_visibility(model.get_value_as_bool())

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
            output_tensor = data.get("output", None) # shape could vary, e.g. [B, C, N] or [B, N, C]
            center = data.get('center', np.zeros(3)) # Default to origin if not saved
            scale = data.get('scale', 1.0)           # Default to 1.0 if not saved

            if input_tensor is None:
                self._inference_label.text = "No 'input' found in PT file."
                return

            # -- Extract input points (x,y,z) from the first 3 channels of 'input'
            # De-normalize for plotting
            points_in_norm = input_tensor[:, :3]
            points_in = input_tensor.detach().cpu()[:, :3]  # [N,3]
            logger.info("[LidarCompletionScenario] Input tensor shape: %s", points_in.shape)
            x_in = points_in[:, 0].numpy()
            y_in = points_in[:, 1].numpy()
            z_in = points_in[:, 2].numpy()

            # -- Extract output points
            x_out, y_out, z_out = None, None, None
            points_out_norm = None
            points_out = None # Initialize points_out
            if output_tensor is not None:
                # Handle potential batch and channel dimensions
                if output_tensor.dim() == 3:
                    if output_tensor.size(1) == 3: # Shape [B, 3, N]
                        points_out_norm = output_tensor.squeeze(0).permute(1, 0) # -> [N, 3]
                    elif output_tensor.size(2) == 3: # Shape [B, N, 3]
                        points_out_norm = output_tensor.squeeze(0) # -> [N, 3]
                elif output_tensor.dim() == 2 and output_tensor.size(1) == 3: # Shape [N, 3]
                    points_out_norm = output_tensor

                points_out = points_out_norm.detach().cpu() # Keep CPU tensor for plotting
                logger.info("[LidarCompletionScenario] Output tensor shape: %s", points_out.shape)
                x_out, y_out, z_out = points_out[:, 0].numpy(), points_out[:, 1].numpy(), points_out[:, 2].numpy()

            # 2) Create a single figure with two subplots, side by side
            fig = plt.figure(figsize=(8,4), dpi=100)

            # Left subplot for Input
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(x_in, y_in, z_in, s=1, c='blue')
            ax1.set_title("Input (Normalized)")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.view_init(elev=30, azim=-60)  # Optional: tweak 3D view angle

            # Right subplot for Output (if available)
            ax2 = fig.add_subplot(122, projection='3d')
            if x_out is not None:
                ax2.scatter(x_out, y_out, z_out, s=1, c='red')
            ax2.set_title("Output (Normalized)" if x_out is not None else "No Output")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            ax2.view_init(elev=30, azim=-60)

            # Optionally unify coordinate scales so both subplots match
            # Combine normalized points for consistent plot scaling
            if points_out_norm is not None:
                combined_norm = torch.cat([points_in_norm, points_out_norm], dim=0).numpy()
            else:
                combined_norm = points_in_norm.numpy()
            min_vals = combined_norm.min(axis=0)
            max_vals = combined_norm.max(axis=0)
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
            self._inference_label.text = f"Error loading/plotting {pt_path}: {e}"
            logger.error(f"Error processing inference result {pt_path}: {e}", exc_info=True)
        finally:
            # Clean up the temporary file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    logger.error(f"Error removing temporary file {temp_path}: {e}")

