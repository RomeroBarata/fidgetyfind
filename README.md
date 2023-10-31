# FidgetyFind
Code for:

[Morais, R., Le, V., Morgan, C., Spittle, A., Badawi, N., Valentine, J., Hurrion, E. M., Dawson, P. A., Tran, T., & Venkatesh, S. (2023). Robust and Interpretable General Movement Assessment Using Fidgety Movement Detection. IEEE Journal of Biomedical and Health Informatics, 27(10), 5042–5053.](https://ieeexplore.ieee.org/document/10195984/)

## Skeleton Detection
Setup the OpenPose detector by following the instructions in the [fine-tuned OpenPose](https://github.com/RomeroBarata/openpose_keras/tree/parallel) model 
for infants. The link is for the `parallel` branch.

After setting everything up, you can run it on your video:
```bash
python -W ignore demo_video.py single_video --video <PATH-TO-YOUR-VIDEO-FILE> --save_root_dir <PATH-TO-SAVE-DIRECTORY>
```

If you have issues with the script using up all your processors, try:
```bash
taskset --cpu-list 0-5 python -W ignore demo_video.py single_video --video <PATH-TO-YOUR-VIDEO-FILE> --save_root_dir <PATH-TO-SAVE-DIRECTORY>
```

## FidgetyFind Environment Setup
Set up the conda environment with:
```bash
conda env create -f environment.yml
conda activate fidgetyfind
```

## FidgetyFind Feature Extraction
With the skeleton extracted and the FidgetyFind conda environment setup and on, run:
```bash
python ./scripts/fidgetyfind-single-video.py --video_filepath <PATH-TO-YOUR-VIDEO-FILE> \
--skeletons_dir <PATH-TO-DIRECTORY-WITH-DETECTED-SKELETONS> --save_root_dir <PATH-TO-DIRECTORY-TO-SAVE-FEATURES>
```

The above script will save FidgetyFind features of the hips, hands, and feet to the specified directory.