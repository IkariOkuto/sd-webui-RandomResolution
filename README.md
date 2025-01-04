# sd-webui-RandomResolution

Generate with randam resolusion for stable diffusion web ui

Notes for Model, Lora, Embeddings, Hypernetworks

Save notes in extensions/sd-webui-simplenote/notes

![Random Resolution screenshot](https://github.com/IkariOkuto/sd-webui-RandomResolution/raw/main/images/rr.png)

## Disclaimer

There are still some areas that have not been fully tested.

It has been confirmed that it can be used normally in the old forge environment.

Use with larger batch sizes has not been tested.

## Installing

This is not extension, but script.

Put randomresolution_script.py in your scripts folder. (Root/scripts)

## Usage

The specified resolution is generated with a probability according to the weight.

For example, if your resolution settings are (512,512,1) and (1024,1024,4), generating images with a batch count of 5 will ideally generate one 512 image and four 1024 images.

Currently, there are six resolution settings, but you can change the maximum value by changing RR_MAXSIZE in the script (probably... untested).