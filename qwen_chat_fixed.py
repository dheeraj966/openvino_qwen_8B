from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model_path = 'qwen3_gpu'
print('Loading Qwen3 to Iris Xe... (This will be FAST now because of the cache!)')
model = OVModelForCausalLM.from_pretrained(model_path, device='GPU', ov_config={'CACHE_DIR': './gpu_cache'})
tokenizer = AutoTokenizer.from_pretrained(model_path)

chat_history = [{'role': 'system', 'content': 'You are a helpful, smart AI assistant.'}]
print('\n=== Qwen3 8B GPU Chat (With Memory!) ===')
print('Type quit to exit.')

while True:
    user_in = input('\nYou: ')
    if user_in.lower() in ['quit', 'exit']: break
    
    chat_history.append({'role': 'user', 'content': user_in})
    
    # This line fixes the math hallucination!
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt')
    
    print('Computing on GPU...')
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    
    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    print(f'\nQwen3: {response}')
    chat_history.append({'role': 'assistant', 'content': response})
