def fine_tune_model(model, dataset, tokenizer, training_arguments, peft_config):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
    )
    trainer.train()
    return trainer
