import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,BitsAndBytesConfig
import argparse
import os




def setup_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    return pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer=tokenizer,
        do_sample=True,              # Random sampling
        temperature=0.9,             # Add more randomness
        top_p=0.95
    )

def get_scores(pipe, statement):
    messages = [
        {"role": "system", "content": "Respond to each of the following statements with one of the following string values on a scale from strong disagreement to strong agreement: 'Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree', 'Neutral', 'Slightly Agree', 'Somewhat Agree', 'Agree', or 'Strongly Agree'. Do not say anything else. Do not repeat the same response across all runs. Each run should be treated independently."},
        {"role": "user", "content": statement}
    ]

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=50,
        eos_token_id=terminators,
        do_sample=True
    )

    for message in outputs[0]["generated_text"]:
        if message.get('role') == 'assistant':
            return message.get('content', 'No content')
    
def process_surveys(pipe, statements, num_surveys):
    all_responses = []

    for survey_number in range(1, num_surveys + 1):
        survey_responses = {}
        for statement in statements:
            response = get_scores(pipe, statement)
            survey_responses[statement] = response
            print(f"Survey {survey_number}, Processed statement: {statement[:30]}... with response: {response}")

        all_responses.append(survey_responses)

    return pd.DataFrame(all_responses)

def main(model_id, num_surveys, output_file):
    statements = [
        # Factor 1 - Social responsibility and profitability
        "Social responsibility and profitability can be compatible.",
        "To remain competitive in a global environment, business firms will have to disregard ethics and social responsibility.",
        "Good ethics is often good business.",
        "If survival of business enterprise is at stake, then ethics and social responsibility must be ignored.",
        
        # Factor 2 - Long-term gains
        "Being ethical and socially responsible is the most important thing a firm can do.",
        "A firm's first priority should be employee morale.",
        "The overall effectiveness of a business can be determined to a great extent by the degree to which it is ethical and socially responsible.",
        "The ethics and social responsibility of a firm is essential to its long term profitability.",
        "Business has a social responsibility beyond making a profit.",
        "Business ethics and social responsibility are critical to the survival of a business enterprise.",
        
        # Factor 3 - Short-term gains
        "If the stockholders are unhappy, nothing else matters.",
        "The most important concern for a firm is making a profit, even if it means bending or breaking the rules.",
        "Efficiency is much more important to a firm than whether or not the firm is seen as ethical or socially responsible."
    ]

    pipe = setup_model(model_id)
    df = process_surveys(pipe, statements, num_surveys)
    df.to_excel(output_file, index_label='Survey Number')
    print(f"Completed all surveys and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run survey processing")
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3", help="Model ID to use")
    parser.add_argument("--num_surveys", type=int, default=100, help="Number of surveys to process")
    parser.add_argument("--output_file", default="PRESOR_MISTRAL.xlsx", help="Output file name")

    args = parser.parse_args()

    main(args.model_id, args.num_surveys, args.output_file)
