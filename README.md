# AIVA-4x500M
![aiva](https://liketowatchblog.files.wordpress.com/2015/10/ex-machina-ava-dress.jpg)

## Introduction
Building Ava from Ex-Machina using Language model paired with vision and audio engines. Using MoE to generated and understand speech, and then using vision models to identify and see physical things, this would work like a near human robot(less bot type and more human type)

Trained models can be downloaded from: [huggingface/aiva-4x500m](https://huggingface.co/shivendrra/avia-4x500m)

## Language Model
Transformer based model, means it has both, encoder and decoder. It uses few new things, like `RMS normalization` same as LLaMa-7b and relational positional encodings for key, query and value in self-attention of encoder layer, `masked attention` uses triangular mask to prevent the model from attending next token in the sequence and `feedforward` uses `GeLU` as an activation function.

`TikToken` is used for tokenization as it has the highest encoding-decoding speed. `p50k_base` is used with `text-davinci-003`.

## Vision Model
It uses SWIN transformer trained to identify faces in a video and check whether a person is speaking or not. It does have CCT, CVT and ViT codes too, but didn't train them. For now it's still work in progress but soon it'll be done.

## Audio Engine
A transformer based model same as OpenAI's Whisper model, not yet test but it should work. I hope so!

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
MIT