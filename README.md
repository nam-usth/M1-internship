# M1 Internship
### USTH-M1-2019-2020


**Author:** 
* HUYNH Vinh Nam [M19.ICT.007]

**Topic:**
* Video classification

## Brief Summary
This repository provides implementation for the "RIVF-2021 - Fast Pornographic Video Detection using Deep Learning"
[Appendix: Known issues](https://github.com/Protossnam/M1_internship#appendix-known-issues)

## Steps
### 1. Install Anaconda
* On Windows: You can find the download link of Anaconda from [here](https://www.anaconda.com/products/individual). Once it's downloaded, execute the installer file and work through the installation steps.
* On Linux:
1. Just simply open your terminal and type:
```
$ cd /tmp
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
$ sha256sum Anaconda3-2020.02-Linux-x86_64.sh
$ bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
2. You can keep pressing ENTER until the end of the license agreement. 

Once you agree to the license, you will be prompted to choose the location of the installation. 

3. You can press ENTER to accept the default location. 

Once installation is complete, the following output should show up:

```
Output
...
installation finished.
Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /home/namhv/.bashrc ? [yes|no]
[no] >>> 
```
Please type ```yes``` to use the ```conda``` command.  

4. After this step, it's time to activate the installation:
```
$ source ~/.bashrc
```

### 2. Set up an environment
* Congratulation, you have the Anaconda ready on your machine, now is the time to clone this repository.
Rename “M1_internship-main” to just “M1_internship”.
* On both Windows and Linux, from the Anaconda Prompt/Terminal:
```
$ conda create --name video_classification python=3.7
$ conda activate video_classification
```

* You should change the current working directory to the cloned folder [M1_internship], then run:
```
(video_classification) $ conda install --yes --file requirements.txt
```

### 3. User manual

#### To be filled

## Appendix: Known issues

#### 1. Error while training with RTX 30 series
I have been reported that the new RTX 30 series comes with the latest CUDA 11 and CuDNN 8 and only compatible with the Tensorflow >= 2.4.0. So for all the older GPU (date-back to the RTX 20 series), it is recommended to follow the ```requirements.txt```. For those who has the RTX 30 series, this repository can only use CPU since the source code relies heavy on TF 1.14 version.