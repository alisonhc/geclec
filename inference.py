import config


def correct_sent(model, tok, sent) -> str:
    input_ids = tok(f"{config.PREFIX}{sent}", return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=200)
    output_sent = tok.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_sent
