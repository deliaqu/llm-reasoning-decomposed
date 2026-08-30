
SYSTEM_PROMPT = "You are a chatbot who is capable of performing the arithmetic problems."

# direct
symbolic_abstraction_direct_instruction = "Please answer the question directly WITHOUT showing the reasoning process. \
At the end, you MUST write the expression with appropriate round brackets after '####' without the units. \
You MUST NOT simplify the expression. You MUST NOT substitute numerical values for variables. Use only the variable symbols provided in the question — do NOT introduce new variable names, but you may use standard mathematical operators and numeric constants as needed."
original_direct_instruction = "Please answer the question directly WITHOUT showing the reasoning process. \
At the end, you MUST write the answer as an integer after '####', without the equation or units."
numerical_abstraction_direct_instruction = "Please answer the question directly without showing the reasoning process. \
At the end, you MUST write the expression or equation with appropriate round brackets after '####', without the units. \
You MUST NOT simplify or evaluate the expression in your final answer. Use only the numbers provided in the question — do NOT introduce variable names, but you may use standard mathematical operators and numeric constants as needed."
computation_direct_instruction = "Please answer the question directly WITHOUT showing the reasoning process. \
At the end, you MUST write the answer as an integer after '####', without the equation."

# CoT
symbolic_abstraction_cot_instruction = "Let's think step by step. \
At the end, you MUST write the expression with appropriate round brackets after '####', without the units. \
You MUST NOT simplify the expression. You MUST NOT substitute numerical values for variables. Use only the variable symbols provided in the question — do NOT introduce new variable names, but you may use standard mathematical operators and numeric constants as needed."
original_cot_instruction = "Let's think step by step. \
At the end, you MUST write the answer as an integer after '####', without the equation or units."
numerical_abstraction_cot_instruction = "Let's think step by step. \
At the end, you MUST write the expression with round brackets after '####' without the units. \
You MUST NOT simplify or evaluate the expression in your final answer. Use only the numbers provided in the question — do NOT introduce variable names, but you may use standard mathematical operators and numeric constants as needed."
computation_cot_instruction = "Let's think step by step. \
At the end, you MUST write the answer as an integer after '####', without the equation."

# Text-answer variants (e.g. phantomwiki: answers are names, dates, or counts).
original_text_direct_instruction = "Please answer the question directly WITHOUT showing the reasoning process. \
At the end, you MUST write the answer after '####'. Write only the answer itself (a name, number, date, or word), with no explanation or units."
original_text_cot_instruction = "Let's think step by step. \
At the end, you MUST write the answer after '####'. Write only the answer itself (a name, number, date, or word), with no explanation or units."

TASK_CONFIG = {
    'original': {
        "question_col": "question",
        "direct": original_direct_instruction,
        "cot": original_cot_instruction,
    },

    'original_text': {
        "question_col": "question",
        "direct": original_text_direct_instruction,
        "cot": original_text_cot_instruction,
    },
    
    'symbolic_abstraction': {
        "question_col": "symbolic_question",
        "direct": symbolic_abstraction_direct_instruction,
        "cot": symbolic_abstraction_cot_instruction,
    },
    
    'numerical_abstraction': {
        "question_col": "question",
        "direct": numerical_abstraction_direct_instruction,
        "cot": numerical_abstraction_cot_instruction,
    },
    
    'arithmetic_computation': {
        "question_col": "arithmetic_question",
        "direct": computation_direct_instruction,
        "cot": computation_cot_instruction,
    }
}