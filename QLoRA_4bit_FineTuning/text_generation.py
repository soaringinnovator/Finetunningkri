def generate_text(model, tokenizer, prompt):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    result = pipe(prompt)
    return result
