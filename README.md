# Official Repository of Online Gaussian Adaptation (OGA) for Vision-Language Models (VLMs)
Authors:
[Clément Fuchs](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Maxime Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.com/citations?user=xb3Zc3cAAAAJ&hl=fr&oi=ao).


<p align="center">
  <img src="images/abstract_barplot_github_version.png" alt="Bar plot" width="700" height="636">
  <br>
  <em>Figure 1. The presented results are averaged over 100 runs. We propose the Expected Tail Accuracy (ETA), i.e., the average over the 10% worst runs, in solid red line. Our method named OGA not only significantly outperforms competitors on average but also has an ETA exceeding their average accuracy on several datasets (e.g., ImageNet and Pets). See our paper [TODO](https://arxiv.org/)</em>
</p>

OGA is an online adaptation method which builds a cache of samples with low zero-shot entropy along a data stream. This cache is then used to build a multivariate Gaussian model of the class conditional likelihoods of the observed features, finally computing updated predictions using a pseudo-bayesian Maximum A Posteriori (MAP) estimator. All details can be found in our paper [TODO](https://arxiv.org/).
The repository also includes a lightweight implementation of [TDA](https://openaccess.thecvf.com/content/CVPR2024/html/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.html) and [DMN](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dual_Memory_Networks_A_Versatile_Adaptation_Approach_for_Vision-Language_Models_CVPR_2024_paper.html) for training free / zero-shot adaptation without test-time augmentations.
# Dependencies
The repository is dependent on [PyTorch](https://pytorch.org/) and [openai-clip](https://pypi.org/project/openai-clip/).
# Datasets
Please follow [DATASETS.md](DATASETS.md) to install the datasets.
You will get a structure with the following dataset names:
```
$DATA/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|–– oxford_flowers/
|–– food-101/
|–– fgvc_aircraft/
|–– sun397/
|–– dtd/
|–– eurosat/
|–– ucf101/
|–– imagenet/
```
# Computing features
main.py needs to have access to cached features and targets. Targets must be encoded as a single integer per sample, corresponding to the class index.
features and targets must be stored at root_cache_path/{dataset}/cache for each dataset.
--root_cache_path defaults to --root_data_path if not precised.
You can use compute_features.py to compute and store features and labels.
Example : 
```bash
python compute_features.py  --data_root_path "E:/DATA" --backbone "vit_b16" --datasets 'sun397' 'imagenet' 'fgvc_aircraft' 'eurosat' 'food101' 'caltech101' 'oxford_pets' 'oxford_flowers' 'stanford_cars' 'dtd' 'ucf101'
```
/!\ Warning: The above command line overwrites previous features for the current architecture.
# Benchmarks

Results presented in our paper can be reproduced using a command line interface. You must specify the root path --data_root_path where you have installed the datasets, as well as the method to benchmark --adapt_method_name (one of "TDA", "DMN" or "OGA"). Results are stored in run_name.json and run_name.pickles files. You can specify a --run_name for the files or let the code generate default names including all necessary information!
The randomness is controlled by the parameters --master_seed and --n_runs. For a same tuple of (master_seed, n_runs), the runs generated (order of samples) are always the same. Note that you may still observe slight variations in results depending on your CUDA and PyTorch version or hardware specifications.
Example :
```bash  
python main.py --data_root_path "E:/DATA" --adapt_method_name "TDA" --datasets 'sun397' 'imagenet' 'fgvc_aircraft' 'eurosat' 'food101' 'caltech101' 'oxford_pets' 'oxford_flowers' 'stanford_cars' 'dtd' 'ucf101'
```


