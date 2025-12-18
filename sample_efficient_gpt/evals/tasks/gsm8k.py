"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from sample_efficient_gpt.evals.tasks.common import Task
from copy import deepcopy


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


LLAMA_SYSTEM_PROMPT = """Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"#### [answer]\" where [answer] is the response to the problem.\n"""


def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    def __init__(self, subset, split, few_shot: bool = False, use_llama_system_prompt: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.use_llama_system_prompt = use_llama_system_prompt
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        self.few_shot_examples = []
        if few_shot:
            # Use first 5 training examples as contextual few-shot pairs
            support_ds = load_dataset("openai/gsm8k", subset, split="train")
            support_ds = support_ds.select(range(min(5, len(support_ds))))
            for row in support_ds:
                question = self._format_question(row["question"])
                answer = row["answer"]
                parts = self._build_answer_parts(answer)
                self.few_shot_examples.append((question, parts))

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def _format_question(self, question):
        if self.use_llama_system_prompt:
            return LLAMA_SYSTEM_PROMPT.format(question=question)
        else:
            return question

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.ds[index]
        question = self._format_question(row["question"])  # string of the question prompt
        answer = row["answer"]  # string of the full solution and the answer after #### marker
        assistant_message_parts = self._build_answer_parts(answer)

        messages = []
        for q, parts in self.few_shot_examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": deepcopy(parts)})

        messages.extend(
            [
                {"role": "user", "content": question},  # note: simple string
                {"role": "assistant", "content": assistant_message_parts},  # note: list of parts (as dicts)
            ]
        )
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response: str):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", "Last message must be from the Assistant"
        # assert isinstance(assistant_message["content"], list), "This is expected to be a list of parts"
        # last_text_part = assistant_message["content"][-1]["text"]  # this contains the final answer in GSM8K
        
        last_text_part = assistant_message["content"].split("\n")[-1]  # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float

    def _build_answer_parts(self, answer: str):
        """Parse GSM8K answer text into chat parts with tool call markers."""
        assistant_message_parts = []
        parts = re.split(r"(<<[^>]+>>)", answer)
        for part in parts:
            if part.startswith("<<") and part.endswith(">>"):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        return "\n".join([x['text'] for x in assistant_message_parts])
