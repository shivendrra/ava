# AVA

![ava](https://liketowatchblog.files.wordpress.com/2015/10/ex-machina-ava-dress.jpg)

## Introduction

Building Ava from Ex-Machina using Language model paired audio engine to generate speech along with a vision model capabale of understanding human emotions &  . Using MoE to generated and understand speech, and then using vision models to identify and see physical things.

Trained models can be downloaded from: [huggingface/ava-v1](https://huggingface.co/shivendrra/ava)

## Call for sponser/donations

This is supposed to be a big experimental project, trying to fuse two different types of data- audio & text togther while making the model feel like interacting to a sentient being.

It needs to train different kinds of model- vision, audio & language (at-least 3 for now) & I've no source of income to fund compute units for this project. If you are interested in this project & rich, feel free to sponser this.

Just add an Issue with tag `sponser` with your contact info or mail me at: shivharsh44@gmail.com

## Language Model

A transformer based language MoE model, using Deepseek's Latent Attention & RoPE for best preformance. Trained over ~20million tokenss(Still in training phase). Has around ~700M params & 2-Experts for now.

## Contribution

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

MIT
