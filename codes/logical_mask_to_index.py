import json
import torch
from typing import Optional

class LogicalMapToIndexMask:
    def __init__(self, logic_map_path: str):
        self.logic_map = json.load(open(logic_map_path))
        
    def mask_for_SAD_query(self, mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to SAD
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="SAD", key="SAD", logic=self.logic_map["SAD"]['SAD'])
        # CEI to SAD
        if consider_CEI:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="SAD", key="CEI", logic=self.logic_map["SAD"]['CEI'])
        # EPrt to SAD
        if consider_empty_prompts:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="SAD", key="EPrt", logic=self.logic_map["SAD"]['EPrt'])
        # Reg to SAD
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="SAD", key="Reg", logic=self.logic_map["SAD"]['Reg'])
        # UReg to SAD
        if consider_uncontrolled_region:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="SAD", key="UReg", logic=self.logic_map["SAD"]['UReg'])
        # Ref to SAD
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="SAD", key="Ref", logic=self.logic_map["SAD"]['Ref'])
        return mask_map
    
    def mask_for_CEI_query(self, mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to CEI
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="CEI", key="SAD", logic=self.logic_map["CEI"]['SAD'])
        # CEI to CEI
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="CEI", key="CEI", logic=self.logic_map["CEI"]['CEI'])
        # EPrt to CEI
        if consider_empty_prompts:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="CEI", key="EPrt", logic=self.logic_map["CEI"]['EPrt'])
        # Reg to CEI
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="CEI", key="Reg", logic=self.logic_map["CEI"]['Reg'])
        # UReg to CEI
        if consider_uncontrolled_region:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="CEI", key="UReg", logic=self.logic_map["CEI"]['UReg'])
        # Ref to CEI
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="CEI", key="Ref", logic=self.logic_map["CEI"]['Ref'])
        return mask_map

    def mask_for_EPrt_query(self, mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to EPrt
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="EPrt", key="SAD", logic=self.logic_map["EPrt"]['SAD'])
        # CEI to EPrt
        if consider_CEI:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="EPrt", key="CEI", logic=self.logic_map["EPrt"]['CEI'])
        # EPrt to EPrt
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="EPrt", key="EPrt", logic=self.logic_map["EPrt"]['EPrt'])
        # Reg to EPrt
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="EPrt", key="Reg", logic=self.logic_map["EPrt"]['Reg'])
        # UReg to EPrt
        if consider_uncontrolled_region:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="EPrt", key="UReg", logic=self.logic_map["EPrt"]['UReg'])
        # Ref to EPrt
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="EPrt", key="Ref", logic=self.logic_map["EPrt"]['Ref'])
        return mask_map
    
    def mask_for_Reg_query(self, mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to Reg
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Reg", key="SAD", logic=self.logic_map["Reg"]['SAD'])
        # CEI to Reg
        if consider_CEI:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Reg", key="CEI", logic=self.logic_map["Reg"]['CEI'])
        # EPrt to Reg
        if consider_empty_prompts:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Reg", key="EPrt", logic=self.logic_map["Reg"]['EPrt'])
        # Reg to Reg
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Reg", key="Reg", logic=self.logic_map["Reg"]['Reg'])
        # UReg to Reg
        if consider_uncontrolled_region:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Reg", key="UReg", logic=self.logic_map["Reg"]['UReg'])
        # Ref to Reg
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Reg", key="Ref", logic=self.logic_map["Reg"]['Ref'])
        return mask_map

    def mask_for_UReg_query(self, mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to UReg
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="UReg", key="SAD", logic=self.logic_map["UReg"]['SAD'])
        # CEI to UReg
        if consider_CEI:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="UReg", key="CEI", logic=self.logic_map["UReg"]['CEI'])
        # EPrt to UReg
        if consider_empty_prompts:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="UReg", key="EPrt", logic=self.logic_map["UReg"]['EPrt'])
        # Reg to UReg
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="UReg", key="Reg", logic=self.logic_map["UReg"]['Reg'])
        # UReg to UReg
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="UReg", key="UReg", logic=self.logic_map["UReg"]['UReg'])
        # Ref to UReg
        mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, ref_num, k_index, query="UReg", key="Ref", logic=self.logic_map["UReg"]['Ref'])
        return mask_map
    
    def mask_for_Ref_query(self, mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region):
        k_index = 0
        # SAD to Ref
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Ref", key="SAD", logic=self.logic_map["Ref"]['SAD'])
        # CEI to Ref
        if consider_CEI:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Ref", key="CEI", logic=self.logic_map["Ref"]['CEI'])
        # EPrt to Ref
        if consider_empty_prompts:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Ref", key="EPrt", logic=self.logic_map["Ref"]['EPrt'])
        # Reg to Ref
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Ref", key="Reg", logic=self.logic_map["Ref"]['Reg'])
        # UReg to Ref
        if consider_uncontrolled_region:
            mask_map, k_index = self.asymmetric_mask(mask_map, q_step, q_index, 1, k_index, query="Ref", key="UReg", logic=self.logic_map["Ref"]['UReg'])
        # Ref to Ref
        mask_map, k_index = self.symmetrical_mask(mask_map, q_step, q_index, k_index, query="Ref", key="Ref", logic=self.logic_map["Ref"]['Ref'])
        return mask_map
    
    def asymmetric_mask(self, mask_map, q_step, q_index, k_step, k_index, query, key, logic: Optional[str] = ['stage', 'reverse stage', 'always', 'never']):
        if logic == "stage":
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, 0] = 0
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, 1] = 1
        elif logic == "reverse stage":
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, 0] = 1
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, 1] = 0
        elif logic == "always":
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, :] = 1
        elif logic == "never":
            mask_map[q_index: q_index + q_step, k_index: k_index + k_step, :] = 0
        else:
            raise ValueError(f"The impact in logic map from {key} to {query} must be one of ['stage', 'reverse stage', 'always', 'never'].")
        k_index += k_step
        return mask_map, k_index
    
    def symmetrical_mask(self, mask_map, q_step, q_index, k_index, query, key, logic: Optional[str] = ['all', 'match', 'match to all', 'all to match', 'never']):
        little_mask = torch.zeros(q_step, q_step, 2)
        if logic == "all":
            little_mask[:, :, :] = 1
        elif logic == "match":
            for i in range(0, q_step):
                little_mask[i, i, :] = 1
        elif logic == "match to all":
            for i in range(0, q_step):
                little_mask[i, i, 0] = 1
            little_mask[:, :, 1] = 1
        elif logic == "all to match":
            little_mask[:, :, 0] = 1
            for i in range(0, q_step):
                little_mask[i, i, 1] = 1
        elif logic == "never":
            little_mask[:, :, :] = 0
        else:
            raise ValueError(f"The impact in logic map from {key} to {query} must be one of ['all', 'match', 'match to all', 'all to match', 'never'].")
        mask_map[q_index: q_index + q_step, k_index: k_index + q_step, :] = little_mask
        k_index += q_step
        return mask_map, k_index

    def __call__(self, 
                 ref_num: int,
                 index_num: int, 
                 split_stages: bool, 
                 consider_CEI: bool,
                 consider_empty_prompts: bool,
                 consider_uncontrolled_region: bool):
        # initialize the mask_map
        if split_stages:
            mask_map = torch.zeros(index_num, index_num, 2)
        else:
            mask_map = torch.zeros(index_num, index_num, 1)

        # Get the mask_map
        # ALL for split_stages: (the logic for did not split stages is not implemented)
        q_index = 0
        if "SAD" in self.logic_map:
            q_step = ref_num
            mask_map = self.mask_for_SAD_query(mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step
        if consider_CEI:
            q_step = 1
            mask_map = self.mask_for_CEI_query(mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step
        if consider_empty_prompts:
            q_step = 1
            mask_map = self.mask_for_EPrt_query(mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step
        if "Reg" in self.logic_map:
            q_step = ref_num
            mask_map = self.mask_for_Reg_query(mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step
        if consider_uncontrolled_region:
            q_step = 1
            mask_map = self.mask_for_UReg_query(mask_map, q_step, q_index, ref_num, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step
        if "Ref" in self.logic_map:
            q_step = ref_num
            mask_map = self.mask_for_Ref_query(mask_map, q_step, q_index, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
            q_index += q_step

        return mask_map


if __name__ == "__main__":
    get_index_mask = LogicalMapToIndexMask(logic_map_path="configs/attention_mask_logic_map.json")
    inputs = json.load(open("structured_inputs/inputs1.json"))
    ref_num = len(inputs) - 1 if "CEI" in inputs else len(inputs)
    split_stages = True
    consider_CEI = True
    consider_empty_prompts = True
    consider_uncontrolled_region = True
    index_num = ref_num + 1 + 1 + ref_num + 1 + ref_num
    mask_map = get_index_mask(ref_num, index_num, split_stages, consider_CEI, consider_empty_prompts, consider_uncontrolled_region)
    print('stage 1:', mask_map[:, :, 0])
    print('stage 2:', mask_map[:, :, 1])