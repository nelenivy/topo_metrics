"""
task_sets.py
Named MTEB task collections for unsupervised metric evaluation.

Task names use **tuples** (not sets) so ``list(TASK_SET_MAP[...])`` has a
**stable, reproducible** order when passed to ``mteb.get_tasks``.
"""

CORE_TASKS = (
    "NFCorpus",
    "SciFact",
    "FiQA2018",
    "Banking77Classification",
    "AmazonCounterfactualClassification",
    "EmotionClassification",
    "STSBenchmark",
    "SprintDuplicateQuestions",
    "AskUbuntuDupQuestions",
)

# Representative benchmark set limited to tasks with <= 100k samples total.
STANDARD_TASKS = (
    "NFCorpus",
    "SciFact",
    "FiQA2018",
    "ArguAna",
    "CQADupstackTexRetrieval",
    "Banking77Classification",
    "AmazonCounterfactualClassification",
    "EmotionClassification",
    "ToxicConversationsClassification",
    "STSBenchmark",
    "SICK-R",
    "STS17",
    "TwitterSemEval2015",
    "AskUbuntuDupQuestions",
    "StackOverflowDupQuestions",
    "TwentyNewsgroupsClustering",
    "SummEval",
)

FULL_BENCHMARK_NAME = "MTEB(eng, v2)"

TASK_SET_MAP = {
    "core": CORE_TASKS,
    "standard": STANDARD_TASKS,
    "full": None,
}
