from transformers import AutoModelForCausalLM, AutoTokenizer

def rewrite_text(content):
    model_name = "sarvamai/sarvam-m"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    # prepare the model input
    prompt = content
    system_prompt = "Rewrite the provided text and keeping the same tone and language, rectifying phonetic, grammatical or orthographic errors if necessary. Strictly DO NOT WRITE ANYTHING ELSE OTHER THAN THE APPROPRIATE ANSWER"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=8192)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    output_text = tokenizer.decode(output_ids)

    if "</think>" in output_text:
        reasoning_content = output_text.split("</think>")[0].rstrip("\n")
        content = output_text.split("</think>")[-1].lstrip("\n").rstrip("</s>")
    else:
        reasoning_content = ""
        content = output_text.rstrip("</s>")

    return content, reasoning_content
