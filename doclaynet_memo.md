

```
고차원이라는 말이 맞나?
이미지의 픽셀적 특성을 사용하지 않았다.
이미지 추출 방식이 필요.
각 이미지 별로 레이아웃이 있으니까 각 이미지 별로 텍스트도 추출하고 이미지도 추출할 수 있지 않을까?
이 문제가 classification? clustering? 목표가 뭐야?

레이아웃 
t-test
군집분석 task 
분류 문제다 군집분석이다.를 정확하게 해야 함.
멀티모달을 

OCR 텍스트 이미지 제대로 하려면 생각할게 많음.

ML + Vision + NLP 
의미에 대한 전략
어떤식의 성능의 편차가 있는지

key: 
텍스트 + 비전을 싹 무시하고 넘어왔다....

워드로만....
이미지의 픽셀
방향은
레이아웃중에 텍스트만 가지고, 이미지만 가지고 임베딩하고,,,,
텍스트 -> 이미지 -> 컨케이트네이션

image downsampling

모델의 목표 
* pdf -> 각 이미지와, 텍스트를 추출해내는 모델을 만들 수 있지 않을까?. 
각 pdf별로 annotation이 있음. 


아래구조를 가진 데이터 셋을 이용해서 어떤 모델을 만들 수 있을까?

json file은 아래처럼 생김: The snippet below shows part of the JSON data for the page shown further above. The text cell shown is the section heading (index: 3).

{
  "metadata": {
    "page_hash": "132a855ee8b23533d8ae69af0049c038171a06ddfcac892c3c6d7e6b4091c642", // unique identifier, equal to filename
    "original_filename": "NASDAQ_FFIN_2002.pdf", // original document filename
    "page_no": 9, // page number in original document
    "num_pages": 28, // total pages in original document
    "original_width": 612, // width in pixels @72 ppi
    "original_height": 792, // height in pixels @72 ppi
    "coco_width": 1025, // with in pixels in PNG and COCO format
    "coco_height": 1025, // with in pixels in PNG and COCO format
    "collection": "ann_reports_00_04_fancy", // sub-collection name
    "doc_category": "financial_reports" // high-level document category
  },
  "cells": [ // all text cells in the digital PDF data
    {
      // Bounding-box coordinates of text cells,
      // formatted as [x,y,w,h] (same as COCO annotations)
      // where (x,y) is the upper-left corner and
      //       (w,h) is the width and height
      // in the coordinate space of (0,0, coco_width, coco_height)
      "bbox": [
        66.99346405228758,
        112.10344760101009,
        290.869358251634,
        13.66279703282828
      ],
      "text": "Leigh Taliaferro, M.D., values consistency.", // string content of cell
      "font": {
        "color": [
          12,
          72,
          142,
          255
        ],
        "name": "/AAAAAC+HelveticaNeue-Medium",
        "size": 1
      }
    },
    ...

Doclaynet extra files 구조.
├── PDF
│   ├── <hash>.pdf
│   ├── ...
├── JSON
│   ├── <hash>.json
│   ├── ...

```
