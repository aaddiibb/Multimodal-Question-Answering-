import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from prompts import Agent_stage1_system_prompt, Agent_stage2_system_prompt, Agent_stage3_system_prompt


# Defining funtion to process images input
def process_image(base_list):
    images = []
    for base in base_list:
        if base == "no_image":
            continue
        try:
            images.append({"type": "image_url", "image_url": {"url": base},})
        except:
            continue
    return images





# #Gemini Keys
# API_KEYS =  ["AIzaSyCr2U6h2ocuGgPHpD7vPlDZMMoG3046F7E",
#              "AIzaSyBrTrxfg0jBm02MEB9kPEVUN2-l6zvm6L4",
#              "AIzaSyDkkIB9jFiX-Cwssr2dYTNhiNKnFPVAW-8",
#              "AIzaSyAALJuLKKPMf50wg8IRbTmRYaO4I2tcYs0",
#              "AIzaSyByoHH19yBnx6e1zmq7N05gvt7EUE35MLk",
#              "AIzaSyA5tDhYn4gcN07-MT4HynlXVed4a9n8rPA",
#              "AIzaSyCNpwhy1jxNeyf9D0v1oKvSiw3wxySY0Qo",
#              "AIzaSyAf8R1KA7vvvPXdEV0kVKQUfIBIpLuUlgk",
#              "AIzaSyDv-JDBRgLmx6wFquT13sOfupY4cPQ2SVU",
#              "AIzaSyCWBb_9o6I-j_VyGtzgp_HdgPZkW7GMCuc",
#              "AIzaSyBE_ikRJZKLoWxe9yJNjsZnQ6rvPzWOqwg"]



# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# client = OpenAI(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/"
# )


client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

# client = OpenAI(api_key=os.getenv("DEEPINFRA_TOKEN"), 
#     base_url="https://api.deepinfra.com/v1/openai")




def get_answer_zs_no_data(client, question, model="gpt-4o-mini"):

    # User input message
    content = [
                {"type": "text", "text": f"Answer the Question: {question}.\n Lets think step by step."}
            ]
    messages = [{"role": "user", "content":content}]

    # Make the API call
    ans = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return {"predicted_answers": ans.choices[0].message.content}




def get_answer_cot(client, question, text, table, images, model="gpt-4o-mini"):
    system_message = {
    "role": "system",
    "content": """You are provided with specific data in the form of text, tables, and images.
    Answer the question using only the provided data and do not incorporate any internal or external knowledge.
    Your output must strictly follow this format:\n
    1. A chain-of-thought reasoning section enclosed between `<reasoning>` and `</reasoning>` tags.\n
    2. A final answer enclosed between `<answer>` and `</answer>` tags.\n\n
    Do not output any text or commentary outside of these tags.
    Let's think step by step."""}

    # User input message
    messages = [system_message]
    content = [
                {"type": "text", "text": f"Here is the Question: {question}\n Here is the text data:\n{text}\n Here is the table data:\n{table}"}
            ]
    final_content = content + process_image(images)
    messages.append(
        {
            "role": "user",
            "content":final_content
        }
    )
    # Make the API call
    ans = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7
    )

    # Return the response content
    return {"predicted_answers": ans.choices[0].message.content}






def get_answer_cot_caption(client, question, text, table, captions, model="gpt-4o-mini"):
    system_message = {
    "role": "system",
    "content": """You are provided with specific data in the form of text, tables, and image(s) descriptions.
    Answer the question using only the provided data and do not incorporate any internal or external knowledge.
    Your output must strictly follow this format:\n
    1. A chain-of-thought reasoning section enclosed between `<reasoning>` and `</reasoning>` tags.\n
    2. A final answer enclosed between `<answer>` and `</answer>` tags.\n\n
    Do not output any text or commentary outside of these tags.
    Let's think step by step."""
    }

    # User input message
    messages = [system_message]
    final_content = [
                {"type": "text", "text": f"Here is the Question: {question}\n Here is the text data:\n{text}\n Here is the table data:\n{table} \n Here are the imges descriptions:\n{captions}"}
            ]
    messages.append(
        {
            "role": "user",
            "content":final_content
        }
    )

    # Make the API call
    ans = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7
    )
    
    # Return the response content
    return {"predicted_answers": ans.choices[0].message.content}








def text_agent(client, question, texts, model="gpt-4o-mini"):
        # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage1_system_prompt}]
    # User input message
    messages.append({"role": "user", "content": f"Here is the text data:\n{texts}\n Here is the question: {question}"})

    # Query the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content





def table_agent(client, question, tables, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage1_system_prompt}]
    # User input message
    messages.append({"role": "user", "content": f"Here is the markdown table data:\n{tables}\n Here is the question: {question}"})

    # Query the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content




def image_agent(client, question, image_urls, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage1_system_prompt}]
    # User input message
    content = [
        {"type": "text", "text": f"Here is the question: {question}"}
    ]
    final_content = content + process_image(image_urls)
    messages.append({
        "role": "user",
        "content": final_content
    })

    # Query the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content




def text_cross_agent(client, question, text_insight, tables, images, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage2_system_prompt}]
    # User input message
    content = [
        {"type": "text", "text": f"Here is the text insight:\n{text_insight}\n Here is the Question: {question}\n Here is the markdown table:\n{tables}"}
    ]
    final_content = content + process_image(images)
    messages.append(
        {
            "role": "user",
            "content": final_content
        }
    )

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content





def table_cross_agent(client, question, text, table_insight, images, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage2_system_prompt}]
    # User input message
    content = [
        {"type": "text", "text": f"Here is the table insights:\n{table_insight} \n\n Here is the Question: {question}\n Here is the text data:\n{text}"}
    ]
    final_content = content + process_image(images)
    messages.append(
        {
            "role": "user",
            "content": final_content
        }
    )
    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content






def image_cross_agent(client, question, text, tables, image_insight, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage2_system_prompt}]

    # User input message
    user_message = {
        "role": "user",
        "content": f"Here is the image insights: \n {image_insight} \n Here is the Question: {question}\n Here is the text data:\n{text}\n Here is the markdown table: \n{tables}\n"
    }

    # Construct the messages with the question and image URLs
    messages.append(user_message)

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content







def reasoning_agent(client, question, text_cross_output, table_cross_output, image_cross_output, model="gpt-4o-mini"):
    # Combine system, text, and user input
    messages = [{"role": "system", "content": Agent_stage3_system_prompt}]

    # User input message
    content = [
        {"type": "text", "text": f"Here is the Question: {question}\n Here is the text cross output:\n{text_cross_output}\n Here is the table cross output:\n{table_cross_output}\n Here is the image cross output:\n{image_cross_output}"}
        # {"type": "text", "text": f"Here is the text cross output:\n{text_cross_output}\n Here is the table cross output:\n{table_cross_output}\n Here is the image cross output:\n{image_cross_output}"}
    ]
    messages.append(
        {
            "role": "user",
            "content": content
        }
    )

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.7,
    )
    return response.choices[0].message.content






# Running the pipeline
def get_answer_MM(client, question, text, tables, images, modal):
    # results = {'Question': question}  # Initialize dictionary with the question
    results = {}

    # Execute the text agent:
    text_insight = text_agent(client, question, text, modal)
    results['Text Agent Output'] = text_insight

    # Execute the table agent:
    table_insight = table_agent(client, question, tables, modal)
    results['Table Agent Output'] = table_insight

    # Execute the image agent:
    image_insight = image_agent(client, question, images, modal)
    results['Image Agent Output'] = image_insight

    # Text cross agent:
    text_cross_output = text_cross_agent(client, question, text_insight, tables, images, modal)
    results['Text Cross Agent Output'] = text_cross_output

    # Table cross agent:
    table_cross_output = table_cross_agent(client, question, text, table_insight, images, modal)
    results['Table Cross Agent Output'] = table_cross_output

    # Image cross agent:
    image_cross_output = image_cross_agent(client, question, text, tables, image_insight, modal)
    results['Image Cross Agent Output'] = image_cross_output

    # Final Reasoning agent:
    answer = reasoning_agent(client, question, text_cross_output, table_cross_output, image_cross_output, modal)
    results['Final Answer'] = answer

    return results


# Running the pipeline
def get_answer_Many(client, question, text, tables, images, modal):
    if  images == ['no_image']:
        images = []
        # results = {'Question': question}
        results = {}

        # Execute the text agent:
        text_insight = text_agent(client, question, text, modal)
        results['Text Agent Output'] = text_insight

        # Execute the table agent:
        table_insight = table_agent(client, question, tables, modal)
        results['Table Agent Output'] = table_insight

        results['Image Agent Output'] = "Not Used"

        # Text cross agent:
        text_cross_output = text_cross_agent(client, question, text_insight, tables, images, modal)
        results['Text Cross Agent Output'] = text_cross_output

        # Table cross agent:
        table_cross_output = table_cross_agent(client, question, text, table_insight, images, modal)
        results['Table Cross Agent Output'] = table_cross_output

        results['Image Cross Agent Output'] = "Not Used"

        # Final Reasoning agent:
        answer = reasoning_agent(client, question, text_cross_output, table_cross_output, "None", modal)
        results['Final Answer'] = answer

        return results
    
    
    elif tables == "No table data":
        # results = {'Question': question}
        results = {}

        # Execute the text agent:
        text_insight = text_agent(client, question, text, modal)
        results['Text Agent Output'] = text_insight

        results['Table Agent Output'] = "Not Used"

        # Execute the image agent:
        image_insight = image_agent(client, question, images, modal)
        results['Image Agent Output'] = image_insight

        # Text cross agent:
        text_cross_output = text_cross_agent(client, question, text_insight, "No table data", images, modal)
        results['Text Cross Agent Output'] = text_cross_output

        results['Table Cross Agent Output'] = "Not Used"

        # Image cross agent:
        image_cross_output = image_cross_agent(client, question, text, "No table data", image_insight, modal)
        results['Image Cross Agent Output'] = image_cross_output

        # Final Reasoning agent:
        answer = reasoning_agent(client, question, text_cross_output, "No table data", image_cross_output, modal)
        results['Final Answer'] = answer

        return results
    
    else:
        return get_answer_MM(client, question, text, tables, images, modal)
    


# Example usage
if __name__ == "__main__":
    from Dataloader import UnifiedQADataLoader
    import argparse
    parser = argparse.ArgumentParser(
        description="Process MultiModalQA data and get responses via the get_answer agent."
    )
    # Files and paths
    parser.add_argument("--dataset_type", type=str, default="multimqa",
                        help="Type of the Data.")
    parser.add_argument("--dev_file", type=str, default="./Datasets/MultiModalQA/Full_Multimodal_dev.jsonl",
                        help="Path to the development JSON file.")
    parser.add_argument("--tables_file", type=str, default="Datasets/MultiModalQA/MMQA_tables.jsonl",
                        help="Path to the tables JSONL file.")
    parser.add_argument("--texts_file", type=str, default="Datasets/MultiModalQA/MMQA_texts.jsonl",
                        help="Path to the texts JSONL file.")
    parser.add_argument("--images_file", type=str, default="Datasets/MultiModalQA/MMQA_images.jsonl",
                        help="Path to the images JSONL file.")
    parser.add_argument("--images_base_url", type=str, default="Datasets/MultiModalQA/final_dataset_images",
                        help="Base URL/path for images dataset.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Name of the model to use for get_answer.")
    parser.add_argument("--results_csv", type=str, default="result.csv",
                        help="Filename for saving the results CSV.")
    parser.add_argument("--errors_csv", type=str, default="errors.csv",
                        help="Filename for saving the errors CSV.")
    parser.add_argument("--num_iterations", type=int, default=12,
                        help="Number of iterations to process.")
    parser.add_argument("--num_threads", type=int, default=50,
                        help="Number of threads to use for concurrent processing.")

    args = parser.parse_args()

    dataloader = UnifiedQADataLoader(
        dataset_type=args.dataset_type,
        dev_file=args.dev_file,
        tables_file=args.tables_file,
        texts_file=args.texts_file,
        images_file=args.images_file,
        images_base_url=args.images_base_url,
        encode_images=True
    )


    agent_inputs = dataloader.get_agent_inputs(1260)
    dummy_state = {
            "question": agent_inputs["question"],
            "type": agent_inputs["type"],
            "modality": agent_inputs["modalities"],
            "true_answers": agent_inputs["answer"],
            "text": agent_inputs["text"],
            "table": agent_inputs["table"],
            "images": agent_inputs["images"],
            "max_depth": 7,
            "client": client,
            "model": "qwen2.5-vl-7b-instruct",
            "confidence_threshold": 0.9
        }
    



    # Pseudo example for the DFS Tree of Thought approach
    question = "What are the key findings from the data?"
    text_data = "Sample text data containing several observations and detailed descriptions."
    table_data = "Column1, Column2\nValue1, Value2"
    image_urls = ["https://example.com/image1.png", "https://example.com/image2.png"]

    # Uncomment the line below to test the full DFS Tree of Thought (requires valid OpenAI API access)
    result = dfs_tree_of_thought(**dummy_state)
    print(result)
    print("true ans:", agent_inputs["answer"])



    # print("Generated new states from the dummy state:")
    # new_states = generate_new_states(dummy_state)
    # for idx, state in enumerate(new_states, start=1):
    #     print(f"\nState {idx}:")
    #     print("Question:", state["question"])
    #     print("Text:", state["text"])
    #     print("Table:", state["table"])
    #     print("Images:", state["images"])
    #     print("Depth:", state["depth"])
