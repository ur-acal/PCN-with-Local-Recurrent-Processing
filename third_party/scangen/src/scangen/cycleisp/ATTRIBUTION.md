# Attribution for CycleISP Components

This subdirectory contains code and concepts adapted from the CycleISP project.

## Original Work

**CycleISP: Real Image Restoration via Improved Data Synthesis**

### Authors
- Syed Waqas Zamir (Inception Institute of Artificial Intelligence)
- Aditya Arora (Australian National University)
- Salman Khan (Australian National University)
- Munawar Hayat (Monash University)
- Fahad Shahbaz Khan (Linköping University)
- Ming-Hsuan Yang (UC Merced & Google Research)
- Ling Shao (Inception Institute of Artificial Intelligence)

### Citation
```bibtex
@inproceedings{zamir2020cycleisp,
  title={CycleISP: Real Image Restoration via Improved Data Synthesis},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2696--2705},
  year={2020}
}
```

### Original Repository
- **GitHub**: https://github.com/swz30/CycleISP
- **Paper**: https://arxiv.org/abs/2003.07761

## License
Components in this directory are used under the Academic Public License from the original CycleISP project. See LICENSE.md for full license terms.

## Modifications
The original CycleISP code has been adapted for ScAN Gen with the following changes:
- Streamlined to focus only on RGB→RAW generation with noise
- Removed denoising and Raw2Rgb components
- Refactored for modern Python package structure
- Added configuration management and CLI interface
- Integrated with Pooch for model management