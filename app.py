import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    model_name = "Salesforce/codegen-2B-multi"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    max_new = model_inputs.get('max_new_tokens', 10)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    generated_ids = model.generate(input_ids, max_new_tokens=max_new)
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Return the results as a dictionary
    return result
