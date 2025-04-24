import sys
import os

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def test_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT().to(device)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL), strict=False)

    # Load and preprocess example images (replace with your own image paths)
    image_dir = sys.argv[1]
    image_names = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    
    images = load_and_preprocess_images(image_names).to(device)


    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx, aggregated_segs_list = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))

        # Predict Tracks
        # choose your own points to track, with shape (N, 2) for one scene
        query_points = torch.FloatTensor([[100.0, 200.0], 
                                            [60.72, 259.94]]).to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

        track = track_list[-1]
        visualize_tracks_on_images(images, track, (conf_score>0.2) & (vis_score>0.2), out_dir="track_visuals")

        # Predict semantic segmantation 
        seg_prediction = model.seg_head(aggregated_tokens_list, images, ps_idx, aggregated_segs_list)
        pred_logits = seg_prediction['pred_logits']
        pred_masks = seg_prediction['pred_masks']
        print('aa')

def test_BA():
    keypoints_np = self.extract_keypoints(first_frame)
    keypoints = torch.from_numpy(keypoints_np).float()
    
    # 3. Track keypoints across all frames
    aggregated_tokens_list, ps_idx = self.model.aggregator(images)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
        track_list, vis_score, conf_score = self.model.track_head(
            aggregated_tokens_list, 
            images, 
            ps_idx, 
            query_points=keypoints[None]
    )
    tracks = track_list[-1]
    
    # 4. Filter tracks based on visibility and confidence
    B, S, K, _ = tracks.shape  # Batch, Sequence, Keypoints
    
    print("Filtering keypoint tracks...")
    self.keypoint_conf_thresh = 0.2
    self.visibility_thresh = 0.2

    valid_track_indices = []
    for i in range(K):
        # Check if this keypoint is valid across all frames
        is_valid = [(conf_score[0, j, i] > self.keypoint_conf_thresh) and 
                    (vis_score[0, j, i] > self.visibility_thresh) 
                    for j in range(S)]
        
        # Only keep tracks that are valid in at least 60% of frames
        if sum(is_valid) >= S * 0.6:
            valid_track_indices.append(i)
    
    filtered_tracks = tracks[0, :, valid_track_indices]

    # 5. Get 3D points for the tracks using the first frame's depth/point map
    world_points = predictions["world_points"].squeeze(0)
    world_points_conf = predictions["world_points_conf"].squeeze(0)
    
    # Get coordinates for the first frame
    first_frame_coords = filtered_tracks[0].long()  # Shape: [N, 2]
    
    # Extract 3D points at these coordinates
    y_coords = first_frame_coords[:, 1].clamp(0, world_points.shape[0]-1)
    x_coords = first_frame_coords[:, 0].clamp(0, world_points.shape[1]-1)

    inlier_mask = torch.ones((N, S), dtype=bool)
    valid_tracks = torch.ones(N, dtype=bool)

    points3D_opt, extrinsics_opt, intrinsics_opt, extra_params_opt, recon = global_BA(
            triangulated_points=triangulated_points, 
            valid_tracks=valid_tracks, 
            pred_tracks=filtered_tracks, 
            inlier_mask=inlier_mask,
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            extra_params=None, 
            image_size=image_size,
            shared_camera=False, 
            camera_type="SIMPLE_PINHOLE"
        )
    
    


if __name__ == '__main__':
    test_inference()
