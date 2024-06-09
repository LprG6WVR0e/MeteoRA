MAX_SAMPLE_THRESHOLD = 100000

MAX_SAMPLE = {
    "abstract_narrative_understanding": MAX_SAMPLE_THRESHOLD,
    "elementary_math_qa": MAX_SAMPLE_THRESHOLD,
    "linguistics_puzzles": MAX_SAMPLE_THRESHOLD,
    "strategyqa": MAX_SAMPLE_THRESHOLD,
    "cnn_dailymail": 1000,
    "formal_fallacies_syllogisms_negation": MAX_SAMPLE_THRESHOLD,
    "logical_deduction": MAX_SAMPLE_THRESHOLD,
    "topical_chat": 1000, 
    "contextual_parametric_knowledge_conflicts": MAX_SAMPLE_THRESHOLD,
    "gsm8k": MAX_SAMPLE_THRESHOLD, 
    "object_counting": MAX_SAMPLE_THRESHOLD,
    "vitaminc_fact_verification": MAX_SAMPLE_THRESHOLD,
    "cs_algorithms": MAX_SAMPLE_THRESHOLD,
    "language_identification": MAX_SAMPLE_THRESHOLD,
    "question_selection": MAX_SAMPLE_THRESHOLD,
    "alpaca": 1000,
    "news_commentary_es": 1000,
    "news_commentary_it": 1000,
    "news_commentary_de": 1000,
    "tracking_shuffled_objects": MAX_SAMPLE_THRESHOLD,
    "goal_step_wikihow": MAX_SAMPLE_THRESHOLD,
    "disfl_qa": MAX_SAMPLE_THRESHOLD,
    "unit_conversion": MAX_SAMPLE_THRESHOLD,
    "paragraph_segmentation": MAX_SAMPLE_THRESHOLD,
    "reasoning_about_colored_objects": MAX_SAMPLE_THRESHOLD,
    "epistemic_reasoning": MAX_SAMPLE_THRESHOLD,
    "play_dialog_same_or_different": MAX_SAMPLE_THRESHOLD,
    "winowhy": MAX_SAMPLE_THRESHOLD,
    "serial_3": MAX_SAMPLE_THRESHOLD,
    "serial_5": MAX_SAMPLE_THRESHOLD,
    "serial_10": MAX_SAMPLE_THRESHOLD,
}

# WARNING: The tasks order in the list will influence MeteoRA model, DO NOT change the order.
METEORA_TASKS = [
    "abstract_narrative_understanding", 
    "alpaca",
    "cnn_dailymail",
    "contextual_parametric_knowledge_conflicts",
    "cs_algorithms",
    "disfl_qa",
    "elementary_math_qa",
    "epistemic_reasoning",
    "formal_fallacies_syllogisms_negation",
    "goal_step_wikihow",
    "gsm8k",
    "language_identification",
    "linguistics_puzzles",
    "logical_deduction",
    "news_commentary_de",
    "news_commentary_es",
    "news_commentary_it",
    "object_counting",
    "paragraph_segmentation",
    "play_dialog_same_or_different",
    "question_selection",
    "reasoning_about_colored_objects",
    "strategyqa",
    "topical_chat",
    "tracking_shuffled_objects",
    "unit_conversion",
    "vitaminc_fact_verification",
    "winowhy",
]