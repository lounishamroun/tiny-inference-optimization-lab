Ok so first I'm experimenting with the function I found linked to the tokenizer object:

tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT,True)
word_ids=hidden_state_token_id['input_ids']

for index,word in enumerate(word_ids):
    print(f' w {word[index]}')
    print(tokenizer._convert_id_to_token(word[index]))

print(f"tokenizer: {type(tokenizer)} | tokenizer: {type(hidden_state_token_id)}")

It works for a specific word id


Ok so I know that private function are not meant to be used outside of the class:

def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

So I will use self.decoder.get instead

tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT,True)
word_ids=hidden_state_token_id['input_ids']

def decode_ids(tokenizer,word_ids):
    word_ids=word_ids.squeeze(0)
    for id_ in word_ids: 
        embedding=tokenizer.decode(id_.item())
        print(embedding)
        