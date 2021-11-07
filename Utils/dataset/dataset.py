from collections import namedtuple
from typing import List

annotation = namedtuple('annotation', 'tag, start, end, ne')


class Dataset:
    def __init__(self):
        self.dataset = []  # храним набор tuple (text, [(annotation, start, end, ne)])
        self.crop_sentences = []

    def load_sentences(self, ann_file_path: str, text_file_path: str):
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()

        with open(ann_file_path, 'r') as ann_file:
            annotations = []
            for line in ann_file.readlines():
                tokens = line.split()
                annotations.append(annotation(tokens[1], tokens[2], tokens[3], tokens[4]))

        cur_tuple = (text, annotations)
        self.dataset.append(cur_tuple)

    def _get_left(self, local_text: str, ann: annotation, window_size) -> List[str]:
        pos = int(ann.start) - 1
        cur_word = ''
        words = []

        while pos >= 0 and len(words) < window_size:
            if local_text[pos] == ' ':
                if cur_word != '':
                    words.append(cur_word[::-1])
                cur_word = ''
            else:
                cur_word += local_text[pos]
            pos -= 1

        if cur_word != '':
            words.append(cur_word[::-1])

        return words[::-1]

    def _get_right(self, local_text: str, ann: annotation, window_size) -> List[str]:
        pos_right = int(ann.end)
        cur_word = ''
        words_right = []

        while pos_right < len(local_text) and len(words_right) < window_size:
            if local_text[pos_right] == ' ':
                if cur_word != '':
                    words_right.append(cur_word)
                cur_word = ''
            else:
                cur_word += local_text[pos_right]

            pos_right += 1

        if cur_word != '':
            words_right.append(cur_word)

        return words_right

    def build_crop_sentences(self, window_size: int):
        for text, annotators in self.dataset:
            local_text = text.replace('\n', '  ')
            local_text = local_text.replace('\r', '  ')
            local_text = local_text.replace('\t', '  ')

            for ann in annotators:
                left = self._get_left(local_text, ann, window_size)
                right = self._get_right(local_text, ann, window_size)

                self.crop_sentences.append((ann, ' '.join(left), ' '.join(right)))