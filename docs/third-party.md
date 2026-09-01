# Third-party voice components

LocalVox keeps third-party code and model licenses separate from the LocalVox application license.

## F5-TTS ONNX

- The upstream F5-TTS code is MIT-licensed.
- The official F5-TTS pretrained weights are licensed CC-BY-NC 4.0 because their training data includes Emilia.
- The DakeQQ F5-TTS-ONNX adapter is Apache-2.0-licensed. LocalVox's worker contains adapted graph-driving and text-preprocessing logic with attribution in the source.
- LocalVox downloads the CPU FP32 ONNX conversion at a pinned revision and treats the converted weights as retaining the original F5-TTS model restrictions. Repository metadata attached to a conversion does not override the upstream weight license.

F5-TTS sources:

- https://github.com/SWivid/F5-TTS
- https://huggingface.co/SWivid/F5-TTS
- https://github.com/DakeQQ/F5-TTS-ONNX
- https://huggingface.co/H5N1AIDS/F5-TTS-ONNX

## OpenVoice V2

OpenVoice code, checkpoints, MeloTTS, MeCab, and dictionary data retain their respective upstream licenses and notices. See the pinned source revisions in `localvox/runtime_installer.py`.
