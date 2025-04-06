import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

import gradio as gr


#model_dir=snapshot_download("./model/Qwen/Qwen2.5-VL-7B-Instruct")
#model_dir = "./model/Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
# default: Load the model on the available device(s)
#建立图片输入和文本输入函数
def input_messages(file_type,file_path,text):
    messages = [
         {
             "role": "user",
             "content": [
                 {
                     "type": file_type,
                     file_type: file_path,
                 },
                 {"type": "text", "text": text},
             ],
         }
     ]
    return messages
#建立模型推理函数
def qwen2describe2image(model_dir,messages,min_pixels,max_pixels):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        #max_memory={0: "13GiB"}
    )


    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
     #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     #    "./model/Qwen/Qwen2.5-VL-7B-Instruct",
     #    torch_dtype=torch.bfloat16,
     #    attn_implementation="flash_attention_2",
      #   device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(model_dir,
                                              min_pixels= min_pixels ,
                                              max_pixels= max_pixels ,
                                              use_fast=True
                                              )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": f"{image}",
    #             },
    #             {"type": "text", "text": f"{text}"},
    #         ],
    #     }
    # ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text



# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        file_type = gr.Dropdown(["image", "video"], label="选择图片或视频")
        text = gr.Textbox(label="图片描述")
#        video = gr.Video(label="上传视频")
        model_dir = gr.Dropdown(choices=["./model/Qwen/Qwen2.5-VL-3B-Instruct-AWQ"], label="选择模型")
        image = gr.Image(type="pil", label="上传图片")
        min_pixels = gr.Slider(minimum=256*28*28, maximum=256*28*28, step=1, label="最小像素")
        max_pixels = gr.Slider(minimum=256*28*28, maximum=256*28*28, step=1, label="最大像素")
        output = gr.Textbox(label="输出结果")

    # 合并按钮点击事件
    btn1 = gr.Button("生成图片描述")
    btn1.click(
        fn=lambda file_type, image, text, model_dir, min_pixels, max_pixels: qwen2describe2image(
            model_dir,
            input_messages(file_type, image, text),
            min_pixels,
            max_pixels
        ),
        inputs=[file_type, image, text, model_dir, min_pixels, max_pixels],
        outputs=output,
    )
    # btn2 = gr.Button("生成视频描述")
    # btn1.click(
    #     fn=lambda file_type, video, text, model_dir, min_pixels, max_pixels: qwen2describe2image(
    #         model_dir,
    #         input_messages(file_type, video, text),
    #         min_pixels,
    #         max_pixels
    #     ),
    #     inputs=[file_type, video, text, model_dir, min_pixels, max_pixels],
    #     outputs=output,
    # )
if __name__ == "__main__":
    demo.launch()


#     print("请输入图片地址")
#     input_image = input()
#     print("请输入图片描述")
#     input_text = input()
#     input_mess = input_messages(input_image,input_text)
#     print("输入的图片地址为：", input_image,"输入的描述为：",input_text)
#     model_dir = "./model/Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
#     qwen2describe2image(model_dir,input_mess)
#     print(qwen2describe2image)