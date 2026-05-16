from allennlp.training.metrics import Metric
from collections import defaultdict
from typing import Dict, Tuple, List


@Metric.register("constituency_parsing_f1_score")
class ConstituencyParsingF1Score(Metric):

    def __init__(self) -> None:
        super(ConstituencyParsingF1Score, self).__init__()
        self.true_positives: Dict[str, int] = defaultdict(int)
        self.false_positives: Dict[str, int] = defaultdict(int)
        self.false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
        self, pred_triples: List[List[Tuple[int, int, str]]], gold_triples: List[List[Tuple[int, int, str]]]
    ) -> None:
        for p_triples, g_triples in zip(pred_triples, gold_triples):
            for label_span in p_triples:
                if label_span in g_triples:
                    self.true_positives[label_span[2]] += 1
                    g_triples.remove(label_span)
                else:
                    self.false_positives[label_span[2]] += 1

            for label_span in g_triples:
                self.false_negatives[label_span[2]] += 1

    def get_metric(self, reset: bool) -> Dict[str, float]:
        # all_tags: Set[str] = set()
        # all_tags.update(self.true_positives.keys())
        # all_tags.update(self.false_positives.keys())
        # all_tags.update(self.false_negatives.keys())
        all_metrics: Dict[str, float] = {}

        # for tag in all_tags:
        #     precision, recall, f1_measure = self._compute_metrics(
        #         self.true_positives[tag], self.false_positives[tag], self.false_negatives[tag]
        #     )
        #     precision_key = "P" + "-" + tag
        #     recall_key = "R" + "-" + tag
        #     f1_key = "F1" + "-" + tag
        #     all_metrics[precision_key] = precision
        #     all_metrics[recall_key] = recall
        #     all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self.true_positives.values()),
            sum(self.false_positives.values()),
            sum(self.false_negatives.values()),
        )
        all_metrics["P"] = precision
        all_metrics["R"] = recall
        all_metrics["F1"] = f1_measure

        if reset:
            self.reset()

        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float, float]:
        precision, recall, f1_measure = 0., 0., 0.
        if true_positives + false_positives != 0:
            precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives != 0:
            recall = true_positives / (true_positives + false_negatives)
        if precision + recall != 0.:
            f1_measure = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1_measure

    def reset(self) -> None:
        self.true_positives: Dict[str, int] = defaultdict(int)
        self.false_positives: Dict[str, int] = defaultdict(int)
        self.false_negatives: Dict[str, int] = defaultdict(int)
