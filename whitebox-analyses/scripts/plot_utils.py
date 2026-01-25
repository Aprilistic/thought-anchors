def get_type2color():
    type2color = {
        "plan_generation": "tab:blue",
        "verbalized_evaluation_awareness": "tab:orange",
        "fact_retrieval": "tab:green",
        "uncertainty_management": "tab:red",
        "result_consolidation": "tab:purple",
        "self_checking": "tab:brown",
    }
    type2color = {
        "plan_generation": "tab:red",
        "verbalized_evaluation_awareness": "tab:green",
        "fact_retrieval": "tab:orange",
        "uncertainty_management": "tab:purple",
        "result_consolidation": "tab:blue",
        "self_checking": "tab:brown",
    }
    return type2color


def get_type2label():
    type2label = {
        "plan_generation": "Plan\nGeneration",
        "verbalized_evaluation_awareness": "Verbalized\nEval\nAwareness",
        "fact_retrieval": "Fact\nRetrieval",
        "uncertainty_management": "Uncertainty\nMgmt.",
        "result_consolidation": "Result\nConsolidation",
        "self_checking": "Self\nChecking",
    }
    return type2label
