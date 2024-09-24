def configure_model(model_name, use_4bit, bnb_4bit_quant_type, device_map):
    compute_dtype = torch.float16 if use_4bit else torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    return model
