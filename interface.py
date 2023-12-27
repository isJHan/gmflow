
import numpy as np
import torch


from main import get_args_parser
from gmflow.gmflow import GMFlow
from utils import frame_utils

ckpts = [
    "/home/jiahan/jiahan/codes/gmflow/pretrained/pretrained/gmflow_chairs-1d776046.pth",
    "/home/jiahan/jiahan/codes/gmflow/pretrained/pretrained/gmflow_kitti-285701a8.pth",
    "/home/jiahan/jiahan/codes/gmflow/pretrained/pretrained/gmflow_sintel-0c07dcb3.pth",
    "/home/jiahan/jiahan/codes/gmflow/pretrained/pretrained/gmflow_things-e9887eda.pth",
]


@torch.no_grad()
def infer(img1_path,
          img2_path,
          resum_ckpt = '/home/jiahan/jiahan/codes/gmflow/pretrained/pretrained/gmflow_sintel-0c07dcb3.pth'
          ):
    parser = get_args_parser()
    args = parser.parse_args(args=[])

    device = 'cuda'
    
    args.inference_dir = "."
    args.output_path = "output"
    args.resume = resum_ckpt
    


    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)
    inference_dir=args.inference_dir
    output_path=args.output_path
    padding_factor=args.padding_factor
    inference_size=args.inference_size
    paired_data=args.dir_paired_data
    save_flo_flow=args.save_flo_flow
    attn_splits_list=args.attn_splits_list
    corr_radius_list=args.corr_radius_list
    prop_radius_list=args.prop_radius_list
    pred_bidir_flow=args.pred_bidir_flow
    fwd_bwd_consistency_check=args.fwd_bwd_consistency_check
    
    print('Load checkpoint: %s' % args.resume)
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(args.resume, map_location=loc)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=args.strict_resume)

    model.eval()
    
    image1 = frame_utils.read_gen(img1_path)
    image2 = frame_utils.read_gen(img2_path)
    
    image1 = np.array(image1).astype(np.uint8)
    image2 = np.array(image2).astype(np.uint8)
    
    if len(image1.shape) == 2:  # gray image, for example, HD1K
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]
        
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    image1, image2 = image1[None].cuda(), image2[None].cuda()
    
    # infer
    results_dict = model(image1, image2,
                     attn_splits_list=attn_splits_list,
                     corr_radius_list=corr_radius_list,
                     prop_radius_list=prop_radius_list,
                     pred_bidir_flow=pred_bidir_flow,
                     )
    
    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
    
    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    
    return flow
