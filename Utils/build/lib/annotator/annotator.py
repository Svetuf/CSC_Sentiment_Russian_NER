from dataset.dataset import Dataset
from abc import ABC, abstractmethod
from tqdm import tqdm


class Model(ABC):
    def __init__(self, model, w2v):
        self.model = model
        self.w2v = w2v

    @abstractmethod
    def __call__(self, s: str) -> float:
        pass


class Annotator:
    def __init__(self, model: Model):
        self.model = model

    def annotate(self, dataset: Dataset, window_size: int):
        data = []
        dataset.build_crop_sentences(window_size)

        for crop_sentence in tqdm(dataset.crop_sentences):
            s = [crop_sentence[1] + ' ' + crop_sentence[0].ne + ' ' + crop_sentence[2]]
            pos_prob = self.model(s)
            data.append((crop_sentence[1], crop_sentence[0].ne, crop_sentence[2], pos_prob))
        self.data = data
