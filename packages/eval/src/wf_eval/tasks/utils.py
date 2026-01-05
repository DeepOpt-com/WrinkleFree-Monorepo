"""Utility functions for custom lm-eval tasks."""


def doc_to_text_summarization(doc: dict) -> str:
    """Format CNN/DailyMail article for summarization prompt.

    Uses a simple, clear prompt format that works well with instruction-tuned
    and base models alike.
    """
    article = doc["article"]
    # Truncate very long articles to avoid context length issues
    max_article_length = 2000
    if len(article) > max_article_length:
        article = article[:max_article_length] + "..."

    return f"""Article: {article}

Summarize the above article in a few sentences:"""


def strip_whitespace(text: str) -> str:
    """Remove leading/trailing whitespace from generated text."""
    if isinstance(text, str):
        return text.strip()
    return text


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE scores using HuggingFace evaluate.

    Returns dict with rouge1, rouge2, rougeL scores.
    """
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predictions, references=references)
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
        }
    except Exception as e:
        # Fallback to rouge-score library
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)

        return {key: sum(vals) / len(vals) for key, vals in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute BLEU score using sacrebleu or nltk."""
    try:
        import evaluate
        bleu = evaluate.load("bleu")
        # BLEU expects references as list of lists
        refs = [[ref] for ref in references]
        result = bleu.compute(predictions=predictions, references=refs)
        return result["bleu"]
    except Exception:
        # Fallback to nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        # Tokenize
        pred_tokens = [p.split() for p in predictions]
        ref_tokens = [[r.split()] for r in references]

        smoothing = SmoothingFunction().method1
        return corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
