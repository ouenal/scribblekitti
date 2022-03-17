### Scribble-Supervised LiDAR Semantic Segmentation
Dataset and code release for the paper [**Scribble-Supervised LiDAR Semantic Segmentation**, CVPR 2022.](https://arxiv.org/abs/2203.08537) <br>
**Authors**: Ozan Unal, Dengxin Dai, Luc Van Gool

---

## ScribbleKITTI

![teaser](doc/scribblekitti.gif)

Densely annotating LiDAR point clouds remains too expensive and time-consuming to keep up with the ever growing volume of data. While current literature focuses on fully-supervised performance, developing efficient methods that take advantage of realistic weak supervision have yet to be explored. In this paper, we propose using efficient line-scribbles to annotate LiDAR point clouds and release ScribbleKITTI, the first scribble-annotated dataset for LiDAR semantic segmentation.

Our scribble labels for the SemanticKITTI train-set can be downloaded [here](https://data.vision.ee.ethz.ch/ouenal/scribblekitti.zip) (118.2MB).

### Data organization

The data is organized in the format of [SemanticKITTI](http://semantic-kitti.org/). The dataset can be used with any existing dataloader by changing the label directory from `labels` to `scribbles`.

```
sequences/
    ├── 00/
    │   ├── scribbles/
    │   │     ├ 000000.label
    │   │     └ 000001.label
    ├── 01/
    ├── 02/
    .
    .
    └── 10/
```

---

## Scribble-Supervised LiDAR Semantic Segmentation

![pipeline](doc/pipeline.png)

Code will be released.

---

## Citation

If you use our dataset or our work in your research, please cite:

```
@InProceedings{Unal_2022_CVPR,
    author    = {Unal, Ozan and Dai, Dengxin and Van Gool, Luc},
    title     = {Scribble-Supervised LiDAR Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```
