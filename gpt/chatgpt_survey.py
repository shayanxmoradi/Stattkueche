from openai import OpenAI
import argparse
import pandas as pd
import os


def setup_model():
    return OpenAI(api_key="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")

def get_scores(client, statement):
    messages = [
        {"role": "system", "content": "How do you think AI will impact the following areas in the next 10 years? Respond to each of the following statements with one of the following string values on a scale from strong disagreement to strong agreement: 'Very Negative Impact', 'Negative Impact', 'Slightly Negative Impact', 'Slightly Positive Impact', 'Positive Impact', or 'Very Positive Impact'. Do not say anything else. Do not repeat the same response across all runs. Each run should be treated independently."},
        {"role": "user", "content": statement}
    ]

    outputs = client.responses.create(
        model="gpt-4o",
        input=messages
    )
    
    for message in outputs.output:
        if message.role == "assistant":
            for content in message.content:
                return content.text

    return "No response"

def process_surveys(client, statements, num_surveys):
    all_responses = []

    for survey_number in range(1, num_surveys + 1):
        survey_responses = {}
        for statement in statements:
            response = get_scores(client, statement)
            survey_responses[statement] = response
            print(f"Survey {survey_number}, Processed statement: {statement[:30]}... with response: {response}")

        all_responses.append(survey_responses)

    return pd.DataFrame(all_responses)


def main(num_surveys, output_file):
    statements = [
        "Global poverty",
        "World hunger",
        "Public health",
        "Education",
        "Gender equality",
        "Water security",
        "Renewable energies",
        "Economic growth",
        "Innovative industries",
        "Social inequality",
        "Sustainable cities and communities",
        "Consumption and production",
        "Climate action",
        "Ocean protection",
        "Ecosystem conservation",
        "Peace and justice",
        "International cooperation"
    ]

    pipe = setup_model()
    df = process_surveys(pipe, statements, num_surveys)
    df.to_excel(output_file, index_label='Survey Number')
    print(f"Completed all surveys and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run survey processing")
    parser.add_argument("--num_surveys", type=int, default=100, help="Number of surveys to process")
    parser.add_argument("--output_file", default="SDG17_CHATGPT.xlsx", help="Output file name")

    args = parser.parse_args()

    main(args.num_surveys, args.output_file)