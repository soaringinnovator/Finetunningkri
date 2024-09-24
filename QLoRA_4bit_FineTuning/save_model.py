def save_trained_model(trainer, new_model_name):
    trainer.model.save_pretrained(new_model_name)
