"""moving_mnist dataset."""

import tensorflow_datasets as tfds
from pathlib import *
_DESCRIPTION = """
# Moving MNIST Dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```

```
"""

_CITATION = """
@article{saxena2021clockwork,
      title={Clockwork Variational Autoencoders}, 
      author={Vaibhav Saxena and Jimmy Ba and Danijar Hafner},
      year={2021},
      eprint={2102.09532},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DOWNLOAD_URL = "https://archive.org/download/moving_mnist/moving_mnist_2digit.zip"


class MovingMnist_2digit(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Moving MNIST dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:#包含数据集的基本信息和每个数据样本的格式.
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                dict(video=tfds.features.Video(shape=(None, 64, 64, 1)))
            ),
            supervised_keys=None,
            homepage="https://archive.org/details/moving_mnist",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager): #数据集的下载和划分.
        """Returns SplitGenerators."""
        #path = dl_manager.download_and_extract(_DOWNLOAD_URL) #传本地数据集的话改这里就可以
        path = Path('/home/guohr/datasets/moving_mnist_2digit')

        return dict(train=self._generate_examples(path /"train-seq100"),
                    test=self._generate_examples(path /"test-seq1000"))

    def _generate_examples(self, path): #每个 split 的生成器, 每次生成一个样本
        """Yields examples."""
        for f in path.glob("*.mp4"):
            yield str(f), dict(video=str(f.resolve()))