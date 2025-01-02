import os
import sys
sys.path.append('./Amphion') # For importing modules relative to the Amphion directory
from Amphion.models.vc.vevo.vevo_utils import *
from huggingface_hub import snapshot_download

def vevo_voice(content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=content_wav_path,
        src_text=None,
        style_ref_wav_path=reference_wav_path,
        timbre_ref_wav_path=reference_wav_path,
    )
    save_audio(gen_audio, output_path=output_path)

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # Content Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq32/*"],
    )
    content_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
    )

    # Content-Style Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # Autoregressive Transformer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
    )
    ar_cfg_path = "./config/Vq32ToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

    # Flow Matching Transformer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = "./config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # Vocoder
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./Amphion/models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Inference
    inference_pipeline = VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    content_wav_path = "./Amphion/models/vc/vevo/wav/source.wav"
    reference_wav_path = "./Amphion/models/vc/vevo/wav/arabic_male.wav"
    output_path = "./Amphion/models/vc/vevo/wav/output_vevovoice.wav"

    vevo_voice(content_wav_path, reference_wav_path, output_path)

    print('done')