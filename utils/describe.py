# from transformers import pipeline
# captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
# print(captioner(pil_image)[0]["generated_text"])

__model__ = None
__feature_extractor__ = None
__tokenizer__ = None
__device__ = None

__blip_captioner__ = None
__llava_captioner__ = None


def init_gpt2():

    global __model__, __feature_extractor__, __tokenizer__, __device__

    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch

    __model__ = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    __feature_extractor__ = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    __tokenizer__ = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    __device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    __model__.to(__device__)


def get_blip():
    from transformers import pipeline

    global __blip_captioner__
    if __blip_captioner__ is None:
        __blip_captioner__ = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-large"
        )
    return __blip_captioner__


def get_llava():
    # https://pytorch.org/get-started/locally/

    from transformers import pipeline
    import torch
    from transformers import BitsAndBytesConfig

    global __llava_captioner__

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    if __llava_captioner__ is None:
        model_id = "llava-hf/llava-1.5-7b-hf"

        __llava_captioner__ = pipeline(
            "image-to-text",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
        )

    return __llava_captioner__


max_length = 30
num_beams = 10
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def describe_image_gpt2(rgb_image, *argv, **kwargs):
    global __model__, __feature_extractor__, __tokenizer__, __device__
    if __model__ is None:
        init_gpt2()
    images = [rgb_image]

    pixel_values = __feature_extractor__(
        images=images, return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(__device__)

    output_ids = __model__.generate(pixel_values, **gen_kwargs)

    preds = __tokenizer__.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def describe_image_blip(pil_image, *argv, **kwargs):
    return get_blip()(pil_image)[0]["generated_text"]


def describe_image_llava(pil_image, prompt=""):
    llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    outputs = get_llava()(
        pil_image, prompt=llava_prompt, generate_kwargs={"max_new_tokens": 200}
    )
    return outputs[0]["generated_text"].split("ASSISTANT: ", 1)[1].strip()
