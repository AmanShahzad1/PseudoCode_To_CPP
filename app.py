import streamlit as st
import torch
import sentencepiece as spm
import math

# Load the trained tokenizer
sp = spm.SentencePieceProcessor(model_file="spoc_tokenizer.model")

# Special tokens
sos_token = sp.piece_to_id("<s>")
eos_token = sp.piece_to_id("</s>")
pad_token = 23999  # Set padding token to 23999 for consistency

# Transformer Model
class TransformerSeq2Seq(torch.nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output.permute(1, 0, 2))

# Positional Encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Load the trained model
model_path = "transformer_seq2seq.pth"
vocab_size = 24000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSeq2Seq(vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inference function
def generate_code(pseudocode):
    input_tokens = sp.encode(pseudocode, out_type=int)
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_tensor, input_tensor[:, :-1])
    output_tokens = output.argmax(dim=-1).squeeze().tolist()
    return sp.decode(output_tokens)

# Streamlit app
st.title("Pseudocode to C++ Code Generator")

# Input text area for pseudocode
pseudocode_input = st.text_area("Enter your pseudocode here:")

# Button to generate code
if st.button("Generate C++ Code"):
    if pseudocode_input:
        generated_code = generate_code(pseudocode_input)
        st.text_area("Generated C++ Code:", generated_code, height=300)
    else:
        st.warning("Please enter some pseudocode to generate C++ code.")