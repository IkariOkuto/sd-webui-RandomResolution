# sd-webui-RandomResolution

Generate with randam resolusion for stable diffusion web ui


![Random Resolution screenshot](https://github.com/IkariOkuto/sd-webui-RandomResolution/raw/main/images/rr.png)

## Disclaimer

There are still some areas that have not been fully tested.

It has been confirmed that it can be used normally in the old forge environment.

Use with only one batch sizes has been tested.

## Installing

This is not extension, but script.

Put randomresolution_script.py in your scripts folder. (Root/scripts)

## Usage

The specified resolution is generated with a probability according to the weight.

For example, if your resolution settings are (512,512,1) and (1024,1024,4), generating images with a batch count of 5 will ideally generate one 512 image and four 1024 images.

Currently, there are six resolution settings, but you can change the maximum value by changing RR_MAXSIZE in the script (probably... untested).

Preset information is stored in the extension folder. (Root/extensions/sd-webui-random-resolution, Because I didn't know where it was available...) 