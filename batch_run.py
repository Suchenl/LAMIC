import torch
from codes import FluxLamicPipeline, LamicProcessor
from codes.transformer_flux_lamic import FluxLamicTransformer2DModel
import json
import os
import torchvision
from torchao.quantization import quantize_, int8_weight_only
from tqdm import tqdm
from utils.image_utils import image_grid
from utils.visualize_bbox_mask import overlay_bbox_masks, overlay_bbox_masks_advanced

def main(args):
    torch_dtype = torch.bfloat16

    flux_kontext_transformer_path = args.flux_kontext_transformer_path
    flux_path = args.flux_path
    transformer = FluxLamicTransformer2DModel.from_pretrained(flux_kontext_transformer_path)
    pipe = FluxLamicPipeline.from_pretrained(flux_path,
                                             transformer=transformer,
                                             torch_dtype=torch_dtype)
    
    print("Applying INT8 weight-only quantization to Transformer and Text Encoder...")
    quantize_(pipe.transformer, int8_weight_only())  # Transformer -> INT8 weights
    if hasattr(pipe, "text_encoder_2"):
        quantize_(pipe.text_encoder_2, int8_weight_only())  # Text Encoder -> INT8 weights

    pipe = pipe.to(torch_dtype)
    pipe.to("cuda")

    if args.reduce_memory_usage:
        # reduce memory usage
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    save_folder = args.save_folder
    input_path = args.input_path
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "masks"), exist_ok=True)
    if args.save_bbox_masks:
        os.makedirs(os.path.join(save_folder, "bboxed"), exist_ok=True)

    all_inputs = json.load(open(input_path))
    num_samples = len(all_inputs)
    num_img_per_sample = args.num_img_per_sample
    
    init_seed = args.init_seed
    output_height = args.output_height
    output_width = args.output_width
    first_stage_ratio = args.first_stage_ratio
    second_stage_ratio = args.second_stage_ratio
    logic_map_path = args.logic_map_path
    auto_resize_ref_img = args.auto_resize_ref_img
    fix_ref_img_size = args.fix_ref_img_size
    ref_size_height = args.ref_size_height
    ref_size_width = args.ref_size_width
    resize_output_size_in_advance = args.resize_output_size_in_advance
    lamic_processor = LamicProcessor(logic_map_path, auto_resize_ref_img, fix_ref_img_size, ref_size_height, ref_size_width)
    generator = torch.Generator(device="cuda")

    test_bar = tqdm(range(num_samples), desc="Processing samples")
    for i in test_bar:
        if args.choose_sample is not None:
            if i not in args.choose_sample:
                continue
        if i < args.start_sample:
            continue
        if args.concat_per_sample:
            image_list = []
            bboxed_image_list = []

        inputs = all_inputs[f"sample_{i:03d}"]
        processed_inputs = lamic_processor(inputs, pipe, height=output_height, width=output_width, resize_output_size=resize_output_size_in_advance,
                                            padding_prompts=True, max_sequence_length=512, save_bbox_masks=args.save_bbox_masks)
        # get index mask for two stages
        index_mask = lamic_processor._get_index_mask(inputs=inputs, processed_inputs=processed_inputs)
        # get attention mask for two stages (if there is mitigate_overlapping_region, attention mask 2 must be first generated, because it will not be mitigated)
        attention_mask_2 = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 1], processed_inputs=processed_inputs, stage=2)
        print("indices before mitigate_overlapping_region: ", [len(token_indice) for token_indice in processed_inputs["token_indices"].values()])
        attention_mask_1_before_mitigate = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 0], processed_inputs=processed_inputs, stage=1)
        # mitigate overlapping region
        processed_inputs = lamic_processor._mitigate_overlapping_region(inputs=inputs, processed_inputs=processed_inputs)
        print("indices after mitigate_overlapping_region: ", [index for index in processed_inputs['mitigated_indices'].keys()], [len(region_indices) for region_indices in processed_inputs['mitigated_indices'].values()])
        attention_mask_1 = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 0], processed_inputs=processed_inputs, stage=1)
        save_masks(index_mask, attention_mask_1, attention_mask_2, save_folder=os.path.join(save_folder, "masks"))
        print("difference between attention mask 1 before and after mitigate_overlapping_region: ", torch.mean(torch.abs(attention_mask_1_before_mitigate.to(torch.float32) - attention_mask_1.to(torch.float32))))

        # full attention
        if args.full_attention:
            attention_mask_2 = torch.ones_like(attention_mask_2).to(torch.bool)
            attention_mask_1 = torch.ones_like(attention_mask_1).to(torch.bool)

        # Generate images for each sample
        for j in range(num_img_per_sample):
            generator.manual_seed(init_seed + 2 * j)
            test_bar.set_description(f"Processing sample {i:03d} image {j + 1}")
            image = pipe(
                image=processed_inputs["images"], # list of images
                prompt=processed_inputs["total_prompts"],
                attention_mask1=attention_mask_1, # attention mask for transformer attention
                attention_mask2=attention_mask_2, # attention mask for transformer attention  
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=processed_inputs["output_height"],
                width=processed_inputs["output_width"],
                generator=generator,
                _auto_resize=auto_resize_ref_img,
                height_width_is_adjusted=True,
                first_stage_ratio=inputs["first_stage_ratio"] if "first_stage_ratio" in inputs else first_stage_ratio,
                second_stage_ratio=inputs["second_stage_ratio"] if "second_stage_ratio" in inputs else second_stage_ratio,
                enhance_factors=processed_inputs["enhance_factors"],
            ).images[0]
            # save single image
            image_save_path = os.path.join(save_folder, f"sample_{i:03d}_image_{j +1}.png") 
            image.save(image_save_path)
            print(f"image 'sample_{i:03d}_image_{j + 1}.png' saved in {image_save_path}")
            
            if args.concat_per_sample:
                image_list.append(image)
            
            if isinstance(processed_inputs["bbox_masks_list"], list) and len(processed_inputs["bbox_masks_list"]) > 0:
                print(processed_inputs["bbox_masks_list"])
                bboxed_image = overlay_bbox_masks_advanced(image, processed_inputs["bbox_masks_list"], fill_alpha=128, outline_only=False)
                # save single bboxed image
                bboxed_image_save_path = os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_image_{j + 1}_bboxed.png")
                bboxed_image.save(bboxed_image_save_path)
                print(f"bboxed image 'sample_{i:03d}_image_{j + 1}_bboxed.png' saved in {bboxed_image_save_path}")
                if args.concat_per_sample:
                    bboxed_image_list.append(bboxed_image)

        if args.concat_per_sample:
            image = image_grid(image_list, len(image_list) // 2, 2)
            save_path = os.path.join(save_folder, f"sample_{i:03d}.png")
            image.save(save_path)
            print(f"Save results sample_{i:03d} to: {save_path}")
            for j in range(num_img_per_sample):
                os.remove(os.path.join(save_folder, f"sample_{i:03d}_image_{j + 1}.png"))
            
            if len(bboxed_image_list) > 0:
                bboxed_image = image_grid(bboxed_image_list, len(bboxed_image_list) // 2, 2)
                bboxed_save_path = os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_bboxed.png")
                bboxed_image.save(bboxed_save_path)
                print(f"Save bboxed results sample_{i:03d} to: {bboxed_save_path}")
                for j in range(num_img_per_sample):
                    os.remove(os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_image_{j + 1}_bboxed.png"))  
                    
            del image_list, bboxed_image_list


def save_masks(index_mask, attention_mask1, attention_mask2, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    to_pil = torchvision.transforms.ToPILImage()
    pil_index_mask_stage1 = to_pil(index_mask[:, :, 0])
    pil_index_mask_stage2 = to_pil(index_mask[:, :, 1])
    pil_index_mask_fuse = to_pil(torch.mean(index_mask, dim=-1))
    pil_attention_mask1 = to_pil(attention_mask1.to(torch.float32))
    pil_attention_mask2 = to_pil(attention_mask2.to(torch.float32))
    pil_index_mask_stage1.save(os.path.join(save_folder, "index_mask_stage1.png"))
    pil_index_mask_stage2.save(os.path.join(save_folder, "index_mask_stage2.png"))
    pil_index_mask_fuse.save(os.path.join(save_folder, "index_mask_fuse.png"))
    pil_attention_mask1.save(os.path.join(save_folder, "attention_mask1.png"))
    pil_attention_mask2.save(os.path.join(save_folder, "attention_mask2.png"))

import argparse
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="./gen_datas/Four-Reference")
    parser.add_argument("--input_path", type=str, default="./dataset/structured_inputs/Four-Reference.json")
    parser.add_argument("--first_stage_ratio", type=float, default=0.05)
    parser.add_argument("--num_img_per_sample", type=int, default=4)
    parser.add_argument("--concat_per_sample", type=bool, default=True)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--output_height", type=int, default=1024)
    parser.add_argument("--output_width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--second_stage_ratio", type=float, default=1.0)
    parser.add_argument("--logic_map_path", type=str, default="configs/attention_mask_logic_map.json")
    parser.add_argument("--auto_resize_ref_img", type=bool, default=False)
    parser.add_argument("--fix_ref_img_size", type=bool, default=False)
    parser.add_argument("--ref_size_height", type=int, default=256)
    parser.add_argument("--ref_size_width", type=int, default=256)
    parser.add_argument("--resize_output_size_in_advance", type=bool, default=True)
    parser.add_argument("--save_bbox_masks", type=bool, default=True)
    parser.add_argument("--choose_sample", type=int, nargs="+", default=None)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--full_attention", action="store_true")
    parser.add_argument("--flux_kontext_transformer_path", type=str, default="/mnt/sda/model_weights/FLUX.1-Kontext-dev/transformer/transformer",
                        help="path to the flux kontext transformer, diffuser format")
    parser.add_argument("--flux_path", type=str, default="/mnt/sata/models/FLUX.1-dev",
                        help="path to the flux model, diffuser format")
    parser.add_argument("--reduce_memory_usage", type=bool, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = set_args()
    main(args)