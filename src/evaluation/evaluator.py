from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer


def compute_bleu(references: list, hypotheses: list) -> float:
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(refs, hyps)


def compute_rouge_l(references: list, hypotheses: list) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores)
