# Lucid Sonic Dreams
Lucid Sonic Dreams syncs GAN-generated visuals to music. By default, it uses [NVLabs StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch), with pre-trained models lifted from [Justin Pinkney's consolidated repository](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Custom weights and other GAN architectures can be used as well.

Sample output can be found on [YouTube](https://youtu.be/l-nGC-ve7sI) and [Instagram](https://www.instagram.com/lucidsonicdreams/).

## Installation  
  
This implementation has been teston on Python 3.6 and 3.7. This now uses the PyTorch implementation of StyleGAN2.

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

### Parameters

The Parameters

Now, the parameters can be easily understood by separating them into 7 categories: Initialization, Pulse, Motion, Class, Effects, Video, and Other.

If this is still overwhelming, it's recommended that you start off by tuning speed_fpm, pulse_react, motion_react and class_pitch_react, and build from there. These parameters make the biggest difference.
Initialization

    speed_fpm (Default: 12) - FPM stands for "Frames per Minute". This determines how many images are initialized - the more there are, the faster the visuals morph. If speed_fpm = 0, then only one image is initialized, and that single image reacts to the audio. In this case, there will be no motion during silent parts of the audio.

Pulse Parameters

    pulse_react (Default: 0.5) - The "strength" of the pulse. It is recommended to keep this between 0 and 2.

    pulse_percussive (Default: True) - If True while pulse_harmonic is False, pulse reacts to the audio's percussive elements.

    pulse_harmonic (Default: False) - If True while pulse_percussive is False, pulse reacts to the audio's harmonic elements.

    Note: If both parameters are True or both parameters are False, pulse reacts to the "entire" unaltered audio.

    pulse_audio - Path to a separate audio file to be used to control pulse. This is recommended if you have access to an isolated drum/percussion track. If passed, pulse_percussive and pulse_harmonic are ignored. Note: this parameter is passed when defining the LucidSonicDream object.

Motion Parameters

    motion_react (0.5), motion_percussive (False), motion_harmonic (True), and motion_audio - Simply the "motion" equivalents of the pulse parameters above.
    motion_randomness (Default: 0.5)- Degree of randomness of motion. Higher values will typically prevent the video from cycling through the same visuals repeatedly. Must range from 0 to 1.
    truncation (Default: 1) - Controls the variety of visuals generated. Lower values lead to lower variety. Note: A very low value will usually lead to "jittery" visuals. Must range from 0 to 1.

Class Parameters

(Note: Most of these parameters were heavily inspired by the Deep Music Visualizer project by Matt Siegelman)

    classes - List of at most 12 numerical object labels. If none, 12 labels are selected at random.
    dominant_classes_first (Default: False)- If True, the list passed to "classes" is sorted by prominence in descending order.
    class_pitch_react (Default: 0.5)- Class equivalent of pulse_react and motion_react. It is recommended to keep this between 0 and 2.
    class_smooth_seconds (Default: 1) - Number of seconds spent smoothly interpolating between each class vector. The higher the value, the less "sudden" the change of class.
    class_complexity (Default: 1) - Controls the "complexity" of images generated. Lower values tend to generate more simple and mundane images, while higher values tend to generate more intricate and bizzare objects. It is recommended to keep this between 0 and 1.
    class_shuffle_seconds (Default: None) - Controls the timestamps wherein the mapping of label to note is re-shuffled. This is recommended when the audio used has a limited range of pitches, but you wish for more classes to be shown. If the value passed is a number n, classes are shuffled every n seconds. If the value passed is a list of numbers, these numbers are used as timestamps (in seconds) wherein classes are shuffled.
    class_shuffle_strength (Default: 0.5) - Controls how drastically classes are re-shuffled. Only applies when class_shuffle_seconds is passed. It is recommended to keep this between 0 and 1.
    class_audio - Class equivalent of pulse_audio and motion_audio. Passed when defining the LucidSonicDream object.

Effects Parameters

    contrast_strength (Default: 0.5) - Strength of default contrast effect. It is recommended to keep this between 0 and 1.

    contrast_percussive (Default: True) - If true, contrast reacts to the audio's percussive elements. Must range from 0 to 1.

    contrast_audio - Equivalent of previous "audio" arguments. Passed when defining the LucidSonicDream object.

    Note: If none of these arguments are passed, the contrast effect will not be applied.

    flash_strength (0.5), flash_percussive (True), and flash_audio - Equivalent of the previous three parameters, but for the a "flash" effect. It is recommended to keep these between 0 and 1. If none of these arguments are passed, the flash effect will not be applied.

    custom_effects - List of custom, user-defined effects to apply (See B.4)

Video Parameters

    resolution - Self-explanatory. Low resolutions are recommended for "trial" renders. If none is passed, unaltered high-resolution images will be used.
    start (Default: 0) - Starting timestamp in seconds.
    duration - Video duration in seconds. If none is passed, full duration of audio will be used.
    output_audio - Final output audio of the video. Overwrites audio from "song" parameter if provided (See B.5)
    fps (Default: 43) - Video Frames Per Second.
    save_frames (Default: False) - If true, saved all individual video frames on disk.

Other

    batch_size (Default: 1) - Determines how many vectors are simoultaneously fed to the model. Larger batch sizes are much faster, but cost more GPU memory
