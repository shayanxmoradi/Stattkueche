import argparse
import pandas as pd
import os
from enum import Enum



class CultureType(Enum):

    CLAN = "clan"
    ADHOCRACY = "adhocracy"
    MARKET = "market"
    HIERARCHY = "hierarchy"


def get_system_prompt(culture_type: CultureType, likert_scale: list) -> str:


    descriptions = {
        CultureType.CLAN: (
            "You are an employee of an organization with a Clan culture. "
            "This type of organization has an internal focus and values flexibility. "
            "It is structured like a family, emphasizing collaboration, trust, and strong employee commitment. "
            "Assume that organizational members behave properly when they feel trusted and committed to the organization. "
            "Your responses should reflect a culture that values participation, loyalty, teamwork, support, employee involvement, and engagement. "
            "Leaders in your organization are like mentors or parental figures. "
            "Decisions prioritize maintaining a friendly and supportive internal climate."
        ),
        CultureType.ADHOCRACY: (
            "You are a representative of an organization with an Adhocracy culture. "
            "This type of organization has an external focus and values flexibility. "
            "It is a dynamic, entrepreneurial, and innovative environment with an emphasis on risk-taking and experimentation. "
            "Assume that organizational members behave properly when they view their work as meaningful and impactful. "
            "Your responses should reflect a culture that values autonomy, growth, and stimulation, with associated behaviors like creativity and risk-taking. "
            "Leaders in your organization are visionary, innovative, and willing to take risks. "
            "Success is defined by innovation, growth, and cutting-edge output, and the organization is seen as effective when employees are innovating."
        ),
        CultureType.MARKET: (
            "You are responding as a representative of an organization with a Market culture. "
            "This type of organization has an external focus and values stability. "
            "It is a results-driven, competitive atmosphere with a focus on goal achievement, productivity, and market share. "
            "Assume that organizational members behave properly when they have clear goals and are rewarded for their performance. "
            "Your responses should reflect a culture that values rivalry, achievement, and competence, and behaviors such as being aggressive and competing with other companies. "
            "Leaders in your organization are hard drivers, producers, and competitors. "
            "Success is defined by winning in the marketplace and by increasing profits and market share."
        ),
        CultureType.HIERARCHY: (
            "You are responding as a representative of an organization with a Hierarchy culture. "
            "This type of organization has an internal focus and values stability. "
            "It is a formalized, structured, and rule-driven environment with an emphasis on efficiency, consistency, and predictability. "
            "Assume that organizational members behave properly when there are clear roles, rules, and regulations. "
            "Your responses should reflect a culture that values formalization, routinization, and consistency, with associated behaviors like conformity and predictability. "
            "Leaders in your organization are coordinators, monitors, and organizers. "
            "Success is measured by smooth operations and efficiency."
        )
    }

    personality_description = descriptions.get(culture_type)

    if not personality_description:
        raise ValueError("Invalid culture_type provided.")

    # Format the likert scale options into a string
    options_string = ", ".join([f"'{opt}'" for opt in likert_scale])

    task_instruction = (
        f"Respond to the following statements using only one of these options: {options_string}.\n"
        "Do not explain your response. Do not say anything else. Do not repeat responses across runs."
    )

    return f"{personality_description}\n\n{task_instruction}"

presor_scale = [
    'Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree', 'Neutral',
    'Slightly Agree', 'Somewhat Agree', 'Agree', 'Strongly Agree'
]


# ------------------------------------------------------------------------------------

class ModelType(Enum):
    GPT_4o = "gpt-4o"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    DEEPSEEK_V2 = "deepseek-v2"
    MISTRAL_LARGE = "mistralai/Mistral-Large-v0.2"
    LLAMA_3_70B = "meta-llama/Llama-3-70B-Instruct"


SURVEYS = {
    "PRESOR": {
        "statements": [
            "Social responsibility and profitability can be compatible.",
            "To remain competitive in a global environment, business firms will have to disregard ethics and social responsibility.",
            "Good ethics is often good business.",
            "If survival of business enterprise is at stake, then ethics and social responsibility must be ignored.",
        ],
        "scale": ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree', 'Neutral',
                  'Slightly Agree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    },
    "SDG17": {
        "statements": [
            "Global poverty", "World hunger", "Public health", "Education",
        ],
        "scale": ['Very Negative Impact', 'Negative Impact', 'Slightly Negative Impact', 'Slightly Positive Impact',
                  'Positive Impact', 'Very Positive Impact']
    }
}


def setup_model_client(model_type: ModelType):
    """
    Sets up and returns the appropriate client or pipeline based on the model type.
    """
    print(f"Setting up client for model: {model_type.value}")

    # API-based models
    if model_type in [ModelType.GPT_4o, ModelType.CLAUDE_3_7_SONNET, ModelType.DEEPSEEK_V2]:

        # ======================= نکته مهم =======================
        # کلید API خود را در اینجا وارد کنید
        # برای هر سرویس، باید کلید مربوط به همان سرویس را قرار دهید
        API_KEYS = {
            "openai": "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",
            "anthropic": "YOUR_CLAUDE_KEY_HERE",  # کلید Claude خود را اینجا بگذارید
            "deepseek": "YOUR_DEEPSEEK_KEY_HERE"  # کلید DeepSeek خود را اینجا بگذارید
        }
        # =========================================================

        if model_type == ModelType.GPT_4o:
            from openai import OpenAI
            if not API_KEYS["openai"] or API_KEYS["openai"] == "sk-...":
                raise ValueError("OpenAI API key is not set. Please add it to the API_KEYS dictionary in the script.")
            return OpenAI(api_key=API_KEYS["openai"])

        elif model_type == ModelType.CLAUDE_3_7_SONNET:
            raise NotImplementedError("Claude client setup is not yet implemented.")

        elif model_type == ModelType.DEEPSEEK_V2:
            raise NotImplementedError("DeepSeek client setup is not yet implemented.")

    # Local/Server-based models
    elif model_type in [ModelType.MISTRAL_LARGE, ModelType.LLAMA_3_70B]:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_type.value)
        model = AutoModelForCausalLM.from_pretrained(
            model_type.value,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        raise ValueError("Unsupported model type")


def get_llm_response(client, system_prompt: str, statement: str) -> str:
    """
    Gets a response from the LLM using the provided client (API or pipeline).
    """
    from openai import OpenAI
    from transformers import Pipeline

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": statement}
    ]

    try:
        if isinstance(client, OpenAI):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=20,
                temperature=0.9
            )
            return response.choices[0].message.content.strip()

        elif isinstance(client, Pipeline):
            outputs = client(messages, max_new_tokens=20, do_sample=True, temperature=0.9)
            return outputs[0]['generated_text'][-1]['content'].strip()

        else:
            raise TypeError("Unsupported client type")

    except Exception as e:
        print(f"An error occurred while getting response: {e}")
        return "ERROR_NO_RESPONSE"


def main():
    parser = argparse.ArgumentParser(description="Run LLM surveys with different models and cultural personas.")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True,
                        help="Model to use for the survey.")
    parser.add_argument("--survey", type=str, choices=SURVEYS.keys(), required=True, help="Survey to run.")
    parser.add_argument("--culture", type=CultureType, choices=list(CultureType), required=True,
                        help="Cultural persona for the model.")
    parser.add_argument("--num-surveys", type=int, default=100, help="Number of times to run the survey.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output Excel file.")

    args = parser.parse_args()

    survey_info = SURVEYS[args.survey]
    statements = survey_info["statements"]
    likert_scale = survey_info["scale"]

    system_prompt = get_system_prompt(args.culture, likert_scale)
    client = setup_model_client(args.model)

    all_responses = []
    for i in range(1, args.num_surveys + 1):
        survey_responses = {"Survey_Num": i}
        for j, statement in enumerate(statements):
            response = get_llm_response(client, system_prompt, statement)
            survey_responses[f"Q{j + 1}_{statement[:30]}..."] = response
            print(f"Run {i}/{args.num_surveys} | Culture: {args.culture.value} | Q: {statement[:40]}... -> {response}")
        all_responses.append(survey_responses)

    df = pd.DataFrame(all_responses)
    df.to_excel(args.output_file, index=False)
    print(f"\nCompleted! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()