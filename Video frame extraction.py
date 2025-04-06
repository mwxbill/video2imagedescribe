

import os

import torch
from sympy.codegen.ast import continue_

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2


from gradiodemo import qwen2describe2image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def input_messages(file_path,text):
    messages = [
         {
             "role": "user",
             "content": [
                 {
                     "type": "image",
                     "image": file_path,
                 },
                 {"type": "text", "text": text},
             ],
         }
     ]
    return messages


def split_text(text, max_chars_per_line):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    final_lines = []
    for line in lines:
        for i in range(0, len(line), max_chars_per_line):
            final_lines.append(line[i:i + max_chars_per_line])

    return final_lines


def save_frame_image(frame,image_path,count):
    print("count:",count)

    # 数据预处理
    #frame = cv2.resize(frame, (640, 320))






    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")



    image_name = f"{image_path}{count}.jpg"
    input_message = input_messages(file_path=image_name, text="描述这张图片")

    cv2.putText(frame,time,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame,f"{count}", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(image_name,frame)
    #try:
    describe_image = qwen2describe2image(
           model_dir="./model/Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
           messages=input_message,
           min_pixels=256 * 28 * 28,
           max_pixels=256 * 28 * 28
        )


    separator = ''
    describe_image =separator.join(describe_image)
    print("des",describe_image)

    lines = split_text(describe_image, 20)
    print("lines",lines)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("./fontpath/QingNiao.ttf",30)

    y_text = 200
    x_text =1
    for line in lines:
        draw.text((x_text,y_text),line,font=font,fill=(255,0,255))
        y_text += 40
        print("line",line)


    #draw.text((1,1000),str(f"{describe_image}"),font=font,fill=(255,255,255))
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    #cv2.putText(frame, f"{describe_image}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,)
    image_image = cv2.imwrite(image_name, frame)
    count += 1
    #return count , frame






def image2video(fps,image_fold):
    fps = fps
    fristflag = True
    for i in os.listdir(image_fold):
        filename  = os.path.join(image_fold,i)
        frame = cv2.imread(filename)
        if fristflag :
            fristflag = False
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            image_resize = (frame.shape[1],frame.shape[0])
            video = cv2.VideoWriter("./img/output.mp4",fourcc,fps,image_resize)
        for index in range(fps):
            frame_suitable = cv2.resize(
                frame,
                image_resize,
                interpolation=cv2.INTER_CUBIC
            )
            video.write(frame_suitable)
        video.release()





if __name__ == "__main__":
    frame_frequency = 25


    image_path = "./image_saving/"
    if not os.path.exists(image_path):
        try:
            os.makedirs(image_path)
        except:
            print("创建文件夹失败")
            exit()
    video_name = './img/2.mp4'

    cap = cv2.VideoCapture(video_name)
    success , frame = cap.read()
    count = 0

    while success:

        if count % frame_frequency == 0:

            with ThreadPoolExecutor(max_workers=10) as excutor:
                excutor.submit(
                    save_frame_image(frame,image_path,count))

        success, frame = cap.read()
        count += 1
    folder_name = r"./image_saving"
    image2video(fps=25,image_fold=folder_name)






#    while success:
#        if frame_count % frame_frequency == 0:
#           count ,out_frame =  save_frame_image(frame,image_path,count)



      #  success ,frame = cap.read()
      #  frame_count += 1





