import torch


def evaluate(device, encoder, decoder, input_tensor, max_length=24):
    SOS_token = 0
    EOS_token = 1
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_tokens = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_tokens.append(topi.item())
                # decoded_words.append(tgt_index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_tokens
