from google.cloud import vision
from google.cloud.vision import types
import io
from PIL import Image, ImageDraw
from enum import Enum
import pandas as pd
import numpy as np

import glob
import sys
import os
import os.path
from pdf2image import convert_from_path

!apt-get install poppler-utils

path = 'claim_submission_only'
K = [f for f in glob.glob(path + "**/*.pdf", recursive=True)]
len(K)
for i in range(len(K)):
    pages = convert_from_path(K[i])
    #pages[0].save("./claim_submission_jpeg/claim_submission"+str(i)+".jpg")
    pages[0].save("./claim_submission_jpeg1/"+str(K[i].rsplit( ".", 1 )[ 0 ])+".jpg")
    print(K[i])
    
####   This code replaces all breaks by space and returns all values within BLOCK in the list format.

def get_paragraph(document):
    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
    paragraphs = []
    lines = []

    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para = ""
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        if symbol.property.detected_break.type == breaks.SPACE:
                            line += ' '
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            para += line
                            line = ''
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append(line)
                            para += line
                            line = ''
                paragraphs.append(para)
        return paragraphs

#paragraphs = get_paragraph(document)

#paragraphs

col = ["name of a tag"]

def get_unique_tags_value(paragraphs, tag):
    out=False
    for i,para in enumerate(paragraphs):
        if(para == tag):
            out = True
            break            
    if out:
        return i
    else:
        return "NA"
 df = pd.DataFrame(columns=col)
 
 path = 'claim_submission_jpeg1/claim_submission_only'
image_file = [f for f in glob.glob(path + "**/*.jpg")]

df = pd.DataFrame(columns=col)
A = []
B = []
C = []
for i in range(len(image_file)):
    #B = np.append(B, image_file[i][45:-4])
    id = image_file[i].find("(")
    name = image_file[i][id+1:id+10]
    print(name)
    C = np.append(C, name)
    client = vision.ImageAnnotatorClient()
    with io.open(image_file[i], 'rb') as image_file1:
        content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image=content_image)
    document = response.full_text_annotation
    paragraphs = get_paragraph(document)
    #dict_tag = {}
    #values = col
    #for val in values:
     #   tag = val
     #   j = get_unique_tags_value(paragraphs, tag)
     #   dict_tag[val] = paragraphs[j+1]
    #A = np.append(A, dict_tag)
    for tag in col:
        k = get_unique_tags_value(paragraphs, tag)
        df[tag] = [paragraphs[k+1]] 
    A = np.append(A, df)

df3 = pd.DataFrame(C)
df3.head()

df1 = pd.DataFrame(A)
#df2 = pd.DataFrame(B)
Text = pd.concat([df3, df1], axis=1)
Text.columns = ['Id','Incident Details']

Text.to_csv("csv_path")
