from typing import Set, Tuple, Union
import pandas as pd
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.tatqa_utils import *


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def normalize_answer_for_matching(answer):

    if not answer:
        return answer

    answer = re.sub(r"\$\(([0-9,.]+)\)", r"-\1", answer)

    if isinstance(answer, str) and "%" in answer:
        answer = answer.replace("%", "").strip()

    answer = re.sub(r"million|percent", "", answer)

    answer = re.sub(r"[$,]", "", answer)

    answer = re.sub(r"\s+", " ", answer).strip()

    return answer


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]],
    gold: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[float, float]:

    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)

    if f1 == 1.0:
        return 1.0, f1

    norm_pred = [normalize_answer_for_matching(p) for p in predicted_bags[0]]
    norm_gold = [normalize_answer_for_matching(g) for g in gold_bags[0]]

    try:
        pred_nums = [float(p) for p in norm_pred if p]
        gold_nums = [float(g) for g in norm_gold if g]

        if pred_nums[0] == gold_nums[0]:
            return 1.0, 1.0
    except:
        pass

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    return exact_match, f1


def normalize_multi_span(answer_list):

    if isinstance(answer_list, list):
        return [str(item).strip() for item in answer_list if item]

    if isinstance(answer_list, str):

        and_parts = re.split(r"\s+and\s+", answer_list)

        result = []
        for part in and_parts:

            if re.search(r"\d,\d", part):

                result.append(part.strip())
            else:

                items = re.split(r"\s*,\s*|\s*;\s*|\s*\|\s*", part)
                result.extend([item.strip() for item in items if item.strip()])

        return result

    return [str(answer_list)]


def extract_gold_answers(qa_annotation):
    """
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    """
    answer_type, scale = qa_annotation["answer_type"], qa_annotation["scale"]
    answer_content = qa_annotation["answer"]
    gold_answers = []
    if answer_type in ["multi-span"]:
        gold_answers = normalize_multi_span(answer_content)
    elif answer_type in ["span"]:
        if isinstance(answer_content, list):
            gold_answers = [str(item) for item in answer_content]
        else:
            gold_answers.append(str(answer_content))
    elif answer_type in ["arithmetic"]:
        if isinstance(answer_content, list):
            gold_answers = [str(item) for item in answer_content]
        else:
            gold_answers.append(str(answer_content))
    elif answer_type in ["count"]:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if (
                    "%" in ans_str
                ):  #  has been handled the answer itself is a percentage
                    ans_str = "%.4f" % ans_num
                else:
                    ans_str = "%.4f" % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]


# handle percentage
def add_percent_pred(prediction_strings, pred_scale, pred):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if (
        not pred_scale and "%" not in pred_str and is_number(pred_str)
    ):  # mode only or no pred_scale num only
        pred_str = to_number(pred_str)
        if pred_str is None:
            return prediction_strings
        prediction_strings.append("%.4f" % pred_str)
    return prediction_strings


class TaTQAEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._op_em = 0.0
        self.op_correct_count = {
            "Span-in-text": 0,
            "Cell-in-table": 0,
            "Spans": 0,
            "Sum": 0,
            "Count": 0,
            "Average": 0,
            "Multiplication": 0,
            "Division": 0,
            "Difference": 0,
            "Change ratio": 0,
            "ignore": 0,
        }
        self.op_total_count = {
            "Span-in-text": 0,
            "Cell-in-table": 0,
            "Spans": 0,
            "Sum": 0,
            "Count": 0,
            "Average": 0,
            "Multiplication": 0,
            "Division": 0,
            "Difference": 0,
            "Change ratio": 0,
            "ignore": 0,
        }
        self.scale_correct_count = {
            "": 0,
            "thousand": 0,
            "million": 0,
            "billion": 0,
            "percent": 0,
        }
        self.scale_total_count = {
            "": 0,
            "thousand": 0,
            "million": 0,
            "billion": 0,
            "percent": 0,
        }
        self._count = 0
        self._details = []

    def __call__(
        self,
        ground_truth: dict,
        prediction: Union[str, List],
        pred_scale="",
        pred_span=None,
        gold_span=None,
        pred_op=None,
        gold_op=None,
    ):  # type: ignore
        """
        :param ground_truth:
        :param prediction:
        :param pred_scale:
        :param pred_span:
        :param gold_span:
        :param pred_op:
        :param gold_op:
        :return:
        """
        if pred_op is not None:
            if pred_op == gold_op:
                self.op_correct_count[pred_op] += 1
                self._op_em += 1
            self.op_total_count[gold_op] += 1

        if pred_scale == ground_truth["scale"]:
            self.scale_correct_count[pred_scale] += 1

        self.scale_total_count[ground_truth["scale"]] += 1
        if not prediction:
            exact_match = 0
            f1_score = 0
            span_exact_match = 0
            span_f1_score = 0
        else:
            gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
            if not gold_answer:
                exact_match = 0
                f1_score = 0
                span_exact_match = 0
                span_f1_score = 0
            else:
                ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)

                if gold_scale == pred_scale:
                    self._scale_em += 1
                prediction = (
                    prediction if isinstance(prediction, list) else [prediction]
                )
                prediction_strings = get_answer_str(prediction, pred_scale)
                prediction_strings = add_percent_pred(
                    prediction_strings, pred_scale, prediction
                )
                exact_match, f1_score = metric_max_over_ground_truths(
                    get_metrics, prediction_strings, ground_truth_answer_strings
                )
                if gold_type in ["arithmetic", "count"]:
                    """if gold type equals with arithmetic and count, set the f1_score == exact_match"""
                    f1_score = exact_match
                if not pred_span:
                    span_exact_match = 0
                    span_f1_score = 0
                else:
                    pred_span_strings = get_answer_str(pred_span, "")
                    gold_span_strings = get_answer_str(gold_span, "")
                    span_exact_match, span_f1_score = metric_max_over_ground_truths(
                        get_metrics,
                        pred_span_strings,
                        gold_span_strings,
                    )

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1
        it = {
            **ground_truth,
            **{
                "pred": prediction,
                "pred_scale": pred_scale,
                "em": exact_match,
                "f1": f1_score,
                "pred_span": pred_span,
                "gold_span": gold_span,
                "span_em": span_exact_match,
                "span_f1": span_f1_score,
            },
        }
        self._details.append(it)

    def get_overall_metric(
        self, reset: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        scale_score = self._scale_em / self._count if self._count > 0 else 0
        op_score = self._op_em / self._count if self._count > 0 else 0
        op_em_detail = {
            "Span-in-text": 0,
            "Cell-in-table": 0,
            "Spans": 0,
            "Sum": 0,
            "Count": 0,
            "Average": 0,
            "Multiplication": 0,
            "Division": 0,
            "Difference": 0,
            "Change ratio": 0,
        }
        scale_em_detail = {
            "": 0,
            "thousand": 0,
            "million": 0,
            "billion": 0,
            "percent": 0,
        }
        for k in op_em_detail.keys():
            op_em_detail[k] = (
                self.op_correct_count[k] / self.op_total_count[k]
                if self.op_total_count[k] > 0
                else 0
            )

        for k in scale_em_detail.keys():
            scale_em_detail[k] = (
                self.scale_correct_count[k] / self.scale_total_count[k]
                if self.scale_total_count[k] > 0
                else 0
            )

        if reset:
            self.reset()
        return exact_match, f1_score, scale_score, op_score

    def get_detail_metric(self):
        df = pd.DataFrame(self._details)
        if len(self._details) == 0:
            return None, None
        em_pivot_tab = df.pivot_table(
            index="answer_type", values=["em"], columns=["answer_from"], aggfunc="mean"
        ).fillna(0)

        f1_pivot_tab = df.pivot_table(
            index="answer_type", values=["f1"], columns=["answer_from"], aggfunc="mean"
        ).fillna(0)
        return em_pivot_tab, f1_pivot_tab

    def get_raw_pivot_table(self):
        df = pd.DataFrame(self._details)
        pivot_tab = df.pivot_table(
            index="answer_type", values=["em"], columns=["answer_from"], aggfunc="count"
        ).fillna(0)
        return pivot_tab

    def get_raw(self):
        return self._details

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._op_em = 0.0
        self._count = 0
        self._details = []
        self.op_correct_count = {
            "Span-in-text": 0,
            "Cell-in-table": 0,
            "Spans": 0,
            "Sum": 0,
            "Count": 0,
            "Average": 0,
            "Multiplication": 0,
            "Division": 0,
            "Difference": 0,
            "Change ratio": 0,
            "ignore": 0,
        }
        self.op_total_count = {
            "Span-in-text": 0,
            "Cell-in-table": 0,
            "Spans": 0,
            "Sum": 0,
            "Count": 0,
            "Average": 0,
            "Multiplication": 0,
            "Division": 0,
            "Difference": 0,
            "Change ratio": 0,
            "ignore": 0,
        }
        self.scale_correct_count = {
            "": 0,
            "thousand": 0,
            "million": 0,
            "billion": 0,
            "percent": 0,
        }
        self.scale_total_count = {
            "": 0,
            "thousand": 0,
            "million": 0,
            "billion": 0,
            "percent": 0,
        }

    def __str__(self):
        return f"TaTQAEmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._count})"
