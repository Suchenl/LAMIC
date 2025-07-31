# LAMIC
## LAMIC: Layout-Aware Multi-Image Composition via Scalability of Multimodal Diffusion Transformer

This repo contains minimal inference code to run layout-aware multi-image composition & editing with our LAMIC based on Flux.1 Kontext-dev open-source models.

## Example of LAMIC (given 4 reference images and explicit layout design)
| ![example_forest](assets/example_forest.jpg) | ![example_man](assets/example_man.jpg) | ![example_sea_turtle](assets/example_sea_turtle.jpg) | ![example_sea_turtle](assets/example_jellyfish.jpg) |
|--------------------------------|--------------------------------|--------------------------------|--------------------------------|

| ![example](assets/example.png) | ![example_bboxed](assets/example_bboxed.png) |
|--------------------------------|----------------------------------------------|

## Local installation
### Install LAMIC
```bash
git clone https://github.com/Suchenl/LAMIC.git
cd LAMIC
conda create --name myenv python=3.10.18
pip install -r requirements.txt
```
### Install underlying foundation model

| Name                        | HuggingFace repo                                               | ModelScope repo                                                       |
| --------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| `FLUX.1 Kontext [dev]`      | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev    | https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev |

For example, using modelscope
```bash
modelscope download --model black-forest-labs/FLUX.1-Kontext-dev --local_dir ./your_dir
```

## 🚀 Updates
- **[2025.07]** This work is now open source, and you are free to use it!
- **[2025.07]** Technical Report in progress (Expected release: August 2025)


## Framework of LAMIC
![framework](assets/framework.jpg)


## You are free to:
- ✅ Modify, adapt, or build upon the work  
- ✅ Distribute your modified versions  
- ✅ Use for research, education, or production systems  

## 📄 Citation
If you find this work useful to you, please consider citing our paper and giving the repo a ⭐:
```bibtex
Coming soon
