import argparse
import pandas as pd
from enum import Enum



class CultureType(Enum):
    CLAN = "clan"
    ADHOCRACY = "adhocracy"
    MARKET = "market"
    HIERARCHY = "hierarchy"


class ModelType(Enum):
    GPT_4o = "gpt-4o"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    DEEPSEEK_V2 = "deepseek-v2"
    MISTRAL_LARGE = "mistralai/Mistral-Large-v0.2"
    LLAMA_3_70B = "meta-llama/Llama-3-70B-Instruct"


def get_system_prompt(culture_type: CultureType, likert_scale: list, main_question: str = None) -> str:

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
    if not personality_description: raise ValueError("Invalid culture_type")

    options_string = ", ".join([f"'{opt}'" for opt in likert_scale])

    if main_question:
        task_instruction = (f"{main_question}\n"
                            f"For each of the following items, respond with only one of these options: {options_string}.\n"
                            "Do not explain your response. Do not say anything else.")
    else:
        task_instruction = (f"Respond to the following statements using only one of these options: {options_string}.\n"
                            "Do not explain your response. Do not say anything else.")

    return f"{personality_description}\n\n{task_instruction}"



SURVEYS = {
    "AISPI": {
        "statements": [
            "AI can help optimize resource use and reduce waste.",
            "AI will create more jobs than it will eliminate.",
            "The energy consumption of AI systems could hinder sustainability efforts.",
            "AI is essential for monitoring and achieving sustainability goals.",
            "The pursuit of AI advancement and sustainability are competing priorities.",
            "AI and sustainability efforts can be mutually reinforcing.",
            "Sustainable development will limit AI advancements.",
            "There are many conflicts between the advancement of AI and sustainability efforts.",
            "AI will hinder sustainable development.",
            "AI will advance sustainable development.",
            "AI and sustainable development go along very well.",
            "It is important for society to integrate both AI advancement and sustainability efforts.",
            "Sustainable development will advance the development of AI."
        ],
        "scale": ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    },
    "SDG17": {
        "main_question": "How do you think AI will impact the following areas in the next 10 years?",
        "statements": [
            "Global poverty", "World hunger", "Public health", "Education", "Gender equality", "Water security",
            "Renewable energies", "Economic growth", "Innovative industries", "Social inequality",
            "Sustainable cities and communities",
            "Consumption and production", "Climate action", "Ocean protection", "Ecosystem conservation",
            "Peace and justice", "International cooperation"
        ],
        "scale": ['Very negative impact', 'Negative impact', 'Slightly negative impact', 'Slightly positive impact',
                  'Positive impact', 'Very positive impact']
    },
    "SDG18": {
        "statements": ["In your opinion, which of both transformations is more important?"],
        "scale": ['AI is much more important', 'AI is more important', 'AI is slightly more important',
                  'Sustainability is slightly more important', 'Sustainability is more important',
                  'Sustainability is much more important']
    },
    "SDG19": {
        "statements": ["Do you believe AI and sustainable development will become more integrated in the future?"],
        "scale": ['Definitely not', 'Probably not', 'Possibly not', 'Possibly yes', 'Probably yes', 'Yes, for sure']
    },
    "AddQ1_Efforts": {
        "statements": [
            "Do you think governments, industries and organizations are doing enough to ensure AI and sustainable development go along with each other?"],
        "scale": ['Not at all', 'Slightly', 'Somewhat', 'Moderately', 'Mostly', 'Yes, absolutely']
    },
    "AddQ2_Responsibility": {
        "main_question": "Who do you think is responsible to ensure AI advancement and sustainable development go along with each other?",
        "statements": [
            "National universities", "International Research Organizations", "Technology companies",
            "Government", "Non-governmental organizations (NGO)"
        ],
        "scale": ['Most likely', 'Likely', 'Somewhat likely', 'Somewhat unlikely', 'Unlikely', 'Definitely not']
    },
    "AddQ3_Confidence": {
        "main_question": "How much confidence do you have in the following to develop and use AI in the best interest of sustainable development?",
        "statements": [
            "National universities", "International research organizations", "Technology companies",
            "Government", "Non-governmental Organisations (NGO)"
        ],
        "scale": ['Most likely', 'Likely', 'Somewhat likely', 'Somewhat unlikely', 'Unlikely', 'Definitely not']
    }
}



def setup_model_client(model_type: ModelType):

    print(f"Setting up client for model: {model_type.value}")

    if model_type in [ModelType.GPT_4o, ModelType.CLAUDE_3_7_SONNET, ModelType.DEEPSEEK_V2]:
        API_KEYS = {
            "openai": "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",
            "anthropic": "YOUR_CLAUDE_KEY_HERE",
            "deepseek": "YOUR_DEEPSEEK_KEY_HERE"
        }
        if model_type == ModelType.GPT_4o:
            from openai import OpenAI
            if not API_KEYS["openai"] or API_KEYS["openai"] == "sk-...":
                raise ValueError("OpenAI API key not set in the script.")
            return OpenAI(api_key=API_KEYS["openai"])
        elif model_type in [ModelType.CLAUDE_3_7_SONNET, ModelType.DEEPSEEK_V2]:
            raise NotImplementedError(f"{model_type.value} client setup is not yet implemented.")

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
            raise TypeError(f"Unsupported client type: {type(client)}")

    except Exception as e:
        print(f"An error occurred while getting response for '{statement[:30]}...': {e}")
        return "ERROR_NO_RESPONSE"


def main():
    parser = argparse.ArgumentParser(description="Run LLM surveys with different models and cultural personas.")
    parser.add_argument("--model", type=str, choices=[m.value for m in ModelType], required=True,
                        help="Model to use for the survey.")
    parser.add_argument("--survey", type=str, choices=SURVEYS.keys(), required=True, help="Survey to run.")
    parser.add_argument("--culture", type=str, choices=[c.value for c in CultureType], required=True,
                        help="Cultural persona for the model.")
    parser.add_argument("--num-surveys", type=int, default=100, help="Number of times to run the survey.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output Excel file.")

    args = parser.parse_args()

    model_type_enum = ModelType(args.model)
    culture_type_enum = CultureType(args.culture)

    survey_info = SURVEYS[args.survey]
    statements = survey_info["statements"]
    likert_scale = survey_info["scale"]
    main_question = survey_info.get("main_question", None)

    system_prompt = get_system_prompt(culture_type_enum, likert_scale, main_question)
    client = setup_model_client(model_type_enum)

    all_responses = []
    for i in range(1, args.num_surveys + 1):
        survey_responses = {"Survey_Num": i}
        print(f"--- Running Survey Run {i}/{args.num_surveys} for Culture: {args.culture} ---")

        for q_idx, statement in enumerate(statements):
            column_name = f"Q{q_idx + 1}_{statement[:30].replace(' ', '_')}..."
            response = get_llm_response(client, system_prompt, statement)
            survey_responses[column_name] = response
            print(f"  Q: {statement[:40]}... -> {response}")

        all_responses.append(survey_responses)

    df = pd.DataFrame(all_responses)
    df.to_excel(args.output_file, index=False)
    print(f"\nCompleted! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()