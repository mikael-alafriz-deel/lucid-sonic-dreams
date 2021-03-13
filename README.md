# Lucid Sonic Dreams
Lucid Sonic Dreams syncs GAN-generated visuals to music. By default, it uses [NVLabs StyleGAN2](https://github.com/NVlabs/stylegan2), with pre-trained models lifted from [Justin Pinkney's consolidated repository](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Custom weights and other GAN architectures can be used as well.

Sample output can be found on [YouTube](https://youtu.be/l-nGC-ve7sI) and [Instagram](https://www.instagram.com/lucidsonicdreams/).

## Installation  
  
This implementation has been teston on Python 3.6 and 3.7. As per NVLabs' TensorFlow implementation of StyleGAN2, TensorFlow 1.15 is required. TensorFlow 2.x is not supported.

To install, simply run: 

```pip install lucidsonicdreams```

## Usage

You may refer to the [Lucid Sonic Dreams Tutorial Notebook](https://colab.research.google.com/drive/1Y5i50xSFIuN3V4Md8TB30_GOAtts7RQD?usp=sharing) for full parameter descriptions and sample code templates. A basic visualization snippet is also found below.

### Basic Visualization

```
from lucidsonicdreams import LucidSonicDream


L = LucidSonicDream(song = 'song.mp3',
                    style = 'abstract photos')

L.hallucinate(file_name = 'song.mp4') 
```
