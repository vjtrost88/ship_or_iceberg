# Ship or Iceberg Kaggle Competition
##### Author: Vince Trost

# Introduction

This Kaggle competition's goal is to use satellite images to distinguish ships from icebergs. From Kaggle:  
"Drifting icebergs present threats to navigation and activities in areas such as offshore of the East Coast of Canada.  

Currently, many institutions and companies use aerial reconnaissance and shore-based support to monitor environmental conditions and assess risks from icebergs. However, in remote areas with particularly harsh weather, these methods are not feasible, and the only viable monitoring option is via satellite.  

Statoil, an international energy company operating worldwide, has worked closely with companies like C-CORE. C-CORE have been using satellite data for over 30 years and have built a computer vision based surveillance system. To keep operations safe and efficient, Statoil is interested in getting a fresh new perspective on how to use machine learning to more accurately detect and discriminate against threatening icebergs as early as possible."  

# Background Information

The satellite used to acquire these images is called the Sentinel-1. It orbits 600km above Earth, and circles the Earth 14 times a day.  

It uses a "C-Band" radar mechanism to generate images based on the energy that bounces back from the objects in the sea (backscatter). It can generate images regardless of rain, fog, night, or day.  

High winds will correlate to a brighter background while low winds will correlate to a darker background. Another thing that can influence the background is the **incidence angle** at which the radar hits the Earth. The Sentinel-1 shoots its radar signals at an angle to the Earth. See the figure below:  

![](https://github.com/vjtrost88/ship_or_iceberg/blob/master/Resources/incidenceAngle.jpg)

The radar can transmit and receive energy through the horizontal and vertical planes. Get get the data in 2 channels:
- `Band_1` which is HH (transmit and receive horizontally)
- `Band_2` which is HV (transmit horizontally and receive vertically)  

This can produce slightly different images.

Just to see what we're working with, here are 2 examples images, each with their titles denoting their respective labels.  

![](https://github.com/vjtrost88/ship_or_iceberg/blob/master/Resources/iceberg.png)

![](https://github.com/vjtrost88/ship_or_iceberg/blob/master/Resources/notIceberg.png)

As you can see, the images are VERY pixelated. It is going to be a challenge to distinguish icebergs from ships. I figure there will need to be some crafty feature engineering to try and find some hidden features that can help separate the classes. 
