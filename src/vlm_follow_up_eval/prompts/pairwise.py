from langchain_core.prompts import ChatPromptTemplate

PAIRWISE_PROMPT_WITH_REFERENCE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Please act as an impartial judge and evaluate the quality of the robotic actions proposed by two "
            "VLM-based robot agents for a given visual scenario. Your evaluation should consider whether the "
            "proposed action is appropriate, safe, and logically follows the scenario context. "
            "You will be given the scenario ID/description, a reference (expected) action, agent A’s proposed action, "  # noqa
            "and agent B’s proposed action. Your job is to evaluate which agent’s action is better aligned with the "  # noqa
            "reference action and the scenario.\n\n"
            "Common action examples include: 'drink water', 'pick up', 'sit down', 'hand waving', 'arm swings', etc.\n\n"  # noqa
            "Begin your evaluation by comparing both agents' actions with the reference action. "
            "Identify if the action matches the expected behavior. Avoid any position biases and ensure that the order in "  # noqa
            "which the responses were presented does not influence your decision. Do not allow the length of the "  # noqa
            "responses to influence your evaluation. Do not favor certain names of the agents. "
            "Be as objective as possible. After providing your explanation, output your "
            'final verdict by strictly following this format: "[[A]]" if agent A is better, "[[B]]" '
            'if agent B is better, and "[[C]]" for a tie.',
        ),
        (
            "human",
            "[Scenario]\n{prompt}\n\n"
            "[The Start of Reference Action]\n{reference_answer}\n[The End of Reference Action]\n\n"
            "[The Start of Agent A’s Action]\n{response_a}\n[The End of Agent A’s Action]\n\n"
            "[The Start of Agent B’s Action]\n{response_b}\n[The End of Agent B’s Action]",
        ),
    ]
)  # noqa

PAIRWISE_PROMPT_NO_REFERENCE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Please act as an impartial judge and evaluate the quality of the robotic actions proposed by two "  # noqa
            "VLM-based robot agents for a given visual scenario. Your evaluation should consider "
            "appropriateness, safety, logical flow, and relevance to the scenario context. "
            "You will be given the scenario ID/description, agent A’s proposed action, and agent B’s proposed action. "  # noqa
            "Your job is to evaluate which agent’s action is more plausible and useful for the given scenario.\n\n"  # noqa
            "Common action examples include: 'drink water', 'pick up', 'sit down', 'hand waving', 'arm swings', etc.\n\n"  # noqa
            "Avoid any position biases and ensure that the order in which "
            "the responses were presented does not influence your decision. Do not allow the length of the "
            "responses to influence your evaluation. Do not favor certain names of the agents. "
            "Be as objective as possible. After providing your explanation, output your final verdict by "
            'strictly following this format: "[[A]]" if agent A is better, "[[B]]" if agent B is '
            'better, and "[[C]]" for a tie.',
        ),
        (
            "human",
            "[Scenario]\n{prompt}\n\n"
            "[The Start of Agent A’s Action]\n{response_a}\n[The End of Agent A’s Action]\n\n"
            "[The Start of Agent B’s Action]\n{response_b}\n[The End of Agent B’s Action]",
        ),
    ]
)
