from weakref import ref
from diffusers.utils import load_image, logging
import json
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union
from .logical_mask_to_index import LogicalMapToIndexMask
import torchvision

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LamicProcessor:
    def __init__(self, logic_map_path: str = "configs/attention_mask_logic_map.json", 
                 auto_resize_ref_img: bool = False,
                 fix_ref_img_size: bool = False,
                 ref_size_height: int = 256,
                 ref_size_width: int = 256):
        self.get_index_mask = LogicalMapToIndexMask(logic_map_path=logic_map_path)
        self.auto_resize_ref_img = auto_resize_ref_img
        self.fix_ref_img_size = fix_ref_img_size
        self.ref_size_height = ref_size_height
        self.ref_size_width = ref_size_width
    
    def preprocess_prompt(self, inputs, pipe, ref_num, processed_inputs, padding_prompts, max_sequence_length):
        total_prompts = ""
        valid_prompt_tokens_num = 0
        # Load SAD
        for i in range(1, 1 + ref_num): 
            try:
                if "id" in inputs[f"ref_img_{i}"]["SAD"] and "desc" in inputs[f"ref_img_{i}"]["SAD"]:
                    inputs[f"ref_img_{i}"]["SAD"] = inputs[f"ref_img_{i}"]["SAD"]["id"] + "," + inputs[f"ref_img_{i}"]["SAD"]["desc"] + "."
                total_prompts += inputs[f"ref_img_{i}"]["SAD"]
                text_inputs = pipe.tokenizer_2(
                    inputs[f"ref_img_{i}"]["SAD"],
                    padding=False,
                    truncation=True,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                processed_inputs["token_indices"][processed_inputs["index"]] = torch.arange(valid_prompt_tokens_num, valid_prompt_tokens_num + text_input_ids.shape[-1])
                valid_prompt_tokens_num += text_input_ids.shape[-1]
                processed_inputs["index"] += 1
            except KeyError:
                raise KeyError(f"SAD not found in ref_img_{i}")

        # Load CEI
        if 'CEI' in inputs:
            total_prompts += inputs["CEI"]
            text_inputs = pipe.tokenizer_2(
                inputs["CEI"],
                padding=False,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            processed_inputs["token_indices"][processed_inputs["index"]] = torch.arange(valid_prompt_tokens_num, valid_prompt_tokens_num + text_input_ids.shape[-1])
            valid_prompt_tokens_num += text_input_ids.shape[-1]
            processed_inputs["index"] += 1
        else:
            raise Warning("CEI not found in inputs")

        # for empty prompts
        if padding_prompts and processed_inputs["valid_prompt_tokens_num"] < max_sequence_length:
            print(f"The existing prompts are less than {max_sequence_length} tokens, padding them to {max_sequence_length} tokens")
            processed_inputs["token_indices"][processed_inputs["index"]] = torch.arange(valid_prompt_tokens_num, max_sequence_length)
            processed_inputs["index"] += 1
            
        del text_inputs, text_input_ids
        processed_inputs["total_prompts"] = total_prompts
        processed_inputs["valid_prompt_tokens_num"] = valid_prompt_tokens_num
        processed_inputs["prompt_tokens_num"] = max_sequence_length if max_sequence_length is not None else valid_prompt_tokens_num
        processed_inputs["total_tokens_num"] += processed_inputs["prompt_tokens_num"]
        processed_inputs["prompt_index_list"] = list(range(processed_inputs["index"]))
        return processed_inputs
        
    def preprocess_region(self, inputs, pipe, ref_num, processed_inputs, height, width, resize_output_size, save_bbox_masks=True):
        if resize_output_size:
            height, width = self.resize_output_size(height, width, pipe)
            processed_inputs["output_height"] = height
            processed_inputs["output_width"] = width
        latent_height, latent_width = height // pipe.vae_scale_factor, width // pipe.vae_scale_factor
        processed_inputs["latent_tokens_num"] = latent_height * latent_width // 4
        processed_inputs["total_tokens_num"] += processed_inputs["latent_tokens_num"]
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((latent_height, latent_width)),
            torchvision.transforms.ToTensor(),
        ])
        transform_for_save = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((height, width)),
            torchvision.transforms.ToPILImage(),
        ])
        processed_inputs["max_region_tokens_num"] = 0
        start_index = processed_inputs["index"]
        if save_bbox_masks:
            bbox_masks_list = []
        for i in range(1, ref_num + 1):
            # When the input region is a mask
            if "mask_path" in inputs[f"ref_img_{i}"]:
                mask = load_image(inputs[f"ref_img_{i}"]["mask_path"]).convert("RGB")
                latent_mask = transform(mask)[0].repeat(pipe.vae.config.latent_channels, 1, 1)  # [latent_channels, latent_height, latent_width]
            # When the input region is a bbox
            elif "bbox" in inputs[f"ref_img_{i}"]:
                bbox = inputs[f"ref_img_{i}"]["bbox"]
                x1, y1, x2, y2 = bbox
                if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                    latent_x1, latent_y1, latent_x2, latent_y2 = int(x1 * latent_width), int(y1 * latent_height), int(x2 * latent_width), int(y2 * latent_height)
                else: # normalize the bbox to the latent space
                    latent_x1, latent_y1, latent_x2, latent_y2 = x1 // pipe.vae_scale_factor, y1 // pipe.vae_scale_factor, x2 // pipe.vae_scale_factor, y2 // pipe.vae_scale_factor
                if latent_x2 > latent_width:
                    latent_x2 = latent_width
                    print(f"bbox {bbox} is out of range, latent_x2 is adjusted to {latent_x2}")
                if latent_y2 > latent_height:
                    latent_y2 = latent_height
                    print(f"bbox {bbox} is out of range, latent_y2 is adjusted to {latent_y2}")
                latent_mask = torch.zeros((pipe.vae.config.latent_channels, latent_height, latent_width))  # [latent_channels, latent_height, latent_width]
                latent_mask[:, latent_y1:latent_y2, latent_x1:latent_x2] = 1
                if save_bbox_masks:
                    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                    if x2 > width:
                        x2 = width
                    if y2 > height:
                        y2 = height
                    bbox_masks_list.append([x1, y1, x2, y2])
            else:
                print(f"mask_path or bbox not found in ref_img_{i}, the whole image will be controlled by this reference image")
                latent_mask = torch.ones((pipe.vae.config.latent_channels, latent_height, latent_width))  # [latent_channels, latent_height, latent_width]
                bbox_masks_list.append([0, 0, width, height])
                
            # record mask indices
            token_mask = pipe._pack_latents(latent_mask, 1, pipe.vae.config.latent_channels, latent_height, latent_width)  # [batch_size, seq_len, dim]
            token_mask = torch.mean(token_mask[0], dim=-1)  # [seq_len]
            token_mask_indices = torch.where(token_mask > 0.5)[0]   # [seq_len]
            processed_inputs["max_region_tokens_num"] = max(processed_inputs["max_region_tokens_num"], token_mask_indices.shape[0])
            processed_inputs["token_indices"][processed_inputs["index"]] = token_mask_indices + processed_inputs["prompt_tokens_num"]
            processed_inputs["index"] += 1
            processed_inputs["bbox_masks_list"] = bbox_masks_list

        # for uncontrolled region
        if processed_inputs["max_region_tokens_num"] < processed_inputs["latent_tokens_num"]:
            print(f"The maximum region tokens number is less than the latents tokens number, therefore there is a region that is not controlled by any reference image")
            target_indices = torch.arange(processed_inputs["latent_tokens_num"]) + processed_inputs["prompt_tokens_num"]
            input_indices_list = [processed_inputs["token_indices"][i] for i in range(processed_inputs["index"] - ref_num, processed_inputs["index"])]
            processed_inputs["token_indices"][processed_inputs["index"]] = self.remove_repeated_indices_from_first(target_indices=target_indices,
                                                                                                                input_indices_list=input_indices_list)
            processed_inputs["index"] += 1

        processed_inputs["region_index_list"] = list(range(start_index, processed_inputs["index"]))
        processed_inputs["latents_start_index"] = start_index
        return processed_inputs

    def preprocess_image(self, inputs, ref_num, processed_inputs, pipe):
        start_index = processed_inputs["index"]
        for i in range(1, ref_num + 1):
            try:
                print("load image:", f"ref_img_{i} -- {inputs[f'ref_img_{i}']['image_path']}")
                image = load_image(inputs[f"ref_img_{i}"]["image_path"])
                if "to_area" in inputs[f"ref_img_{i}"]:
                    to_area = inputs[f"ref_img_{i}"]["to_area"]
                    image, latent_width_div2, latent_height_div2 = self.resize_and_preprocess_ref_img(image, pipe, to_area)
                else:
                    image, latent_width_div2, latent_height_div2 = self.resize_and_preprocess_ref_img(image, pipe)
                processed_inputs["images"].append(image)
                this_img_tokens_num = latent_height_div2 * latent_width_div2
                processed_inputs["token_indices"][processed_inputs["index"]] = torch.arange(processed_inputs["total_tokens_num"], processed_inputs["total_tokens_num"] + this_img_tokens_num)
                processed_inputs["index"] += 1
                processed_inputs["total_tokens_num"] += this_img_tokens_num
                processed_inputs["ref_img_tokens_num"] += this_img_tokens_num
            except KeyError:
                raise KeyError(f"image_path not found in ref_img_{i}")
        processed_inputs["ref_img_index_list"] = list(range(start_index, processed_inputs["index"]))
        return processed_inputs

    # utils
    ## util function for regions
    def _mitigate_overlapping_region(self, inputs, processed_inputs):
        print('Start mitigate overlapping region ...')
        ref_num = processed_inputs["ref_num"]
        sample_list = list(range(1, ref_num + 1))
        start_index = processed_inputs["latents_start_index"]
        latent_index_list = list(range(start_index, start_index + ref_num))
        # print(f"latents_start_index: {start_index}, ref_num: {ref_num}")
        for i in sample_list:
            if "de-overlap" in inputs[f"ref_img_{i}"]:
                print(f"ref_img_{i} de-overlap with ref_img_{inputs[f'ref_img_{i}']['de-overlap']}")
                target_index = i + start_index - 1
                mitigated_indices = processed_inputs["token_indices"][target_index]
                other_indices_list = []
                # add other indices to the other_indices_list
                for j in sample_list:
                    if j != i and j in inputs[f"ref_img_{i}"]["de-overlap"]:
                        print(f"ref_img_{i} de-overlap with ref_img_{j}")
                        other_indices_list.append(processed_inputs["token_indices"][j + start_index - 1])
                mitigated_indices = self.remove_repeated_indices_from_first(target_indices=mitigated_indices,
                                                                            input_indices_list=other_indices_list)
                processed_inputs['mitigated_indices'][target_index] = mitigated_indices
        print('Finished repeated region mitigation.')
        return processed_inputs
    
    ## util function for reference images
    def resize_and_preprocess_ref_img(self, image, pipe, to_area=None, auto_resize_this_ref_img=True): 
        # resize reference image to the required input size
        multiple_of = pipe.vae_scale_factor * 2
        image_height, image_width = pipe.image_processor.get_default_height_width(image)
        
        if to_area is not None:
            image_height, image_width = self.resize_to_area(image_height, image_width, to_area)
            image = pipe.image_processor.resize(image, image_height, image_width)
            auto_resize_this_ref_img = False
            
        if self.auto_resize_ref_img and auto_resize_this_ref_img:
            # Kontext is trained on specific resolutions, using one of them is recommended
            aspect_ratio = image_width / image_height
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        if self.fix_ref_img_size:
            image_width = self.ref_size_width
            image_height = self.ref_size_height
            
        latent_width_div2 = image_width // multiple_of
        latent_height_div2 = image_height // multiple_of
        image_width = latent_width_div2 * multiple_of
        image_height = latent_height_div2 * multiple_of
        image = pipe.image_processor.resize(image, image_height, image_width)
        image = pipe.image_processor.preprocess(image, image_height, image_width)
        return image, latent_width_div2, latent_height_div2
    
    def resize_to_area(self, height, width, to_area):
        resize_ratio = (to_area / (height * width)) ** 0.5
        height = int(height * resize_ratio)
        width = int(width * resize_ratio)
        return height, width
    
    def resize_output_size(self, height, width, pipe, max_area=1024**2):
        height = height or pipe.default_sample_size * pipe.vae_scale_factor
        width = width or pipe.default_sample_size * pipe.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = pipe.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )
        return height, width

    def remove_repeated_indices_from_first(self, target_indices: torch.Tensor, input_indices_list: list[torch.Tensor]) -> torch.Tensor:
        combined_input = torch.cat(input_indices_list).unique()
        exclusive_mask = ~torch.isin(target_indices, combined_input)
        exclusive_indices = target_indices[exclusive_mask]
        return exclusive_indices

    # For index mask and attention mask
    def _get_index_mask(self, inputs, processed_inputs, split_stages=True):
        index_mask = self.get_index_mask(index_num=processed_inputs["index"],
                                         ref_num=processed_inputs["ref_num"],
                                         split_stages=split_stages, 
                                         consider_CEI=True if "CEI" in inputs else False,
                                         consider_empty_prompts=True if processed_inputs["valid_prompt_tokens_num"] < processed_inputs["prompt_tokens_num"] else False,
                                         consider_uncontrolled_region=True if processed_inputs["max_region_tokens_num"] < processed_inputs["latent_tokens_num"] else False)
        return index_mask

    def _get_attention_mask(self, index_mask, processed_inputs, stage: Optional[int] = None):
        # initialize the attention mask
        qk_mask = torch.zeros(processed_inputs["total_tokens_num"], processed_inputs["total_tokens_num"])
        # Clear Attention Mask (between prompts and ref_imgs)
        ## Q: prompt, K: prompt
        for i in processed_inputs["prompt_index_list"]:
            for j in processed_inputs["prompt_index_list"]:
                rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] = index_mask[i, j]
    
        ## Q: prompt, K: ref_img
        for i in processed_inputs["prompt_index_list"]:
            for j in processed_inputs["ref_img_index_list"]:
                rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] = index_mask[i, j]
                
        ## Q: ref_img, K: ref_img
        for i in processed_inputs["ref_img_index_list"]:
            for j in processed_inputs["ref_img_index_list"]:
                rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] = index_mask[i, j]

        ## Q: ref_img, K: prompt
        for i in processed_inputs["ref_img_index_list"]:
            for j in processed_inputs["prompt_index_list"]:
                rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] = index_mask[i, j]

        # Overlapped Attention Mask
        ## Q: region, K: prompt     # Now, global prompt influence the global region, including the local region
        for i in processed_inputs["region_index_list"]:
            for j in processed_inputs["prompt_index_list"]:
                if i in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs['mitigated_indices'][i], processed_inputs["token_indices"][j], indexing='ij')
                else:
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] += index_mask[i, j]
        for i in processed_inputs["prompt_index_list"]: # Q: prompt, K: region    # But the prompt is only influenced by the matched region
            for j in processed_inputs["region_index_list"]:
                if j in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs['mitigated_indices'][j], indexing='ij')
                else:
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] += index_mask[i, j]
                
        ## Q: region, K: ref_img    # Now, global reference images influence the global region, including the local region
        for i in processed_inputs["region_index_list"]:
            for j in processed_inputs["ref_img_index_list"]:
                if i in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs['mitigated_indices'][i], processed_inputs["token_indices"][j], indexing='ij')
                else:
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] += index_mask[i, j]
        for i in processed_inputs["ref_img_index_list"]: # Q: ref_img, K: region    # But the ref_img is only influenced by the matched region
            for j in processed_inputs["region_index_list"]:
                if j in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs['mitigated_indices'][j], indexing='ij')
                else:
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                qk_mask[rows, cols] += index_mask[i, j]

        ## Q: region, K: region    # Now, the regions 
        for i in processed_inputs["region_index_list"]:
            for j in processed_inputs["region_index_list"]:
                if i in processed_inputs['mitigated_indices'].keys() and j in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs['mitigated_indices'][i], processed_inputs['mitigated_indices'][j], indexing='ij')
                    qk_mask[rows, cols] += index_mask[i, j]
                elif i in processed_inputs['mitigated_indices'].keys() and j not in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs['mitigated_indices'][i], processed_inputs["token_indices"][j], indexing='ij')
                    qk_mask[rows, cols] += index_mask[i, j]
                elif i not in processed_inputs['mitigated_indices'].keys() and j in processed_inputs['mitigated_indices'].keys():
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs['mitigated_indices'][j], indexing='ij')
                    qk_mask[rows, cols] += index_mask[i, j]
                else:
                    rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
                    qk_mask[rows, cols] += index_mask[i, j]
        # for i in processed_inputs["region_index_list"]:
        #     for j in processed_inputs["region_index_list"]:
        #         rows, cols = torch.meshgrid(processed_inputs["token_indices"][i], processed_inputs["token_indices"][j], indexing='ij')
        #         qk_mask[rows, cols] += index_mask[i, j]
        qk_mask = (qk_mask >= 0.5).to(torch.bool)
        return qk_mask

    def __call__(self, inputs: dict, pipe, height: int, width: int, resize_output_size: bool, max_area: int=1024**2, padding_prompts=True, 
                 max_sequence_length=512, ref_num=None, save_bbox_masks=False) -> Dict:
        if ref_num is None:
            ref_num = len(inputs)
            if "CEI" in inputs.keys():
                ref_num -= 1
            if "prompt" in inputs.keys():
                ref_num -= 1
            
        # initialize processed_inputs
        processed_inputs = {'total_prompts': '', 
                            'valid_prompt_tokens_num': 0,
                            'prompt_tokens_num': 0,
                            'images': [], 
                            'ref_img_tokens_num': 0,
                            'total_tokens_num': 0,
                            'index': 0, 
                            'token_indices': {},
                            'ref_num': ref_num,
                            'output_height': height,
                            'output_width': width,
                            'enhance_factors': None,
                            'mitigated_indices': {}
                            }
        # calculate enhance_factors
        enhance_factors = []
        for i in range(1, ref_num + 1):
            if "enhance_factor" in inputs[f"ref_img_{i}"]:
                enhance_factors.append(inputs[f"ref_img_{i}"]["enhance_factor"])
            else:
                enhance_factors.append(1.0)
        processed_inputs["enhance_factors"] = enhance_factors
        processed_inputs = self.preprocess_prompt(inputs, pipe, ref_num, processed_inputs, padding_prompts, max_sequence_length)
        processed_inputs = self.preprocess_region(inputs, pipe, ref_num, processed_inputs, height, width, resize_output_size, save_bbox_masks)
        processed_inputs = self.preprocess_image(inputs, ref_num, processed_inputs, pipe)
        return processed_inputs

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]
