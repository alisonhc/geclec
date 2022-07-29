import config


def correct_sent(model, tok, sent) -> str:
    input_ids = tok(f"{config.PREFIX}{sent}", return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=200)
    output_sent = tok.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_sent


def correct_many_sents(model, tok, sent_list) -> str:
    output_sents = []
    for sent in sent_list:
        output = correct_sent(model, tok, sent)
        output_sents.append(output)
    return ' '.join(output_sents)