# The virtual personal trainer
>This virtual personal trainer can help you with your rehabilitation 

## Installation

1.Clone and install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2.Clone this repository with:
```sh
git clone https://github.com/jabosso/ElaboratoIVA.git 
```
3. On Ubuntu: you must create a symbolic link inside this folder that points to pyopenpose (library  created after installing OpenPose)

4. Prerequisites for run this code:
    * OpenCV  
    * scipy   
    * numpy 
    * math 
    * argparse 
    * time<br>
   You can install these using pip or Anaconda

## Usage for the user
In the Data folder you can see the types of exercises performed by your personal trainer (you have to unzip the folder to see the videos).
Choose your exercise and record and save your video! 
**Recommendations**:<br> 
    1. Opencv does not read video metadata, a landscape shot is appropriate<br>
    2. Use a quality camera, the webcam is not recommended
    
# You are curious to know how you performed the exercise?
To run the code insert to command line, inside the directory containing the files:
```sh
python match.py -v >Path_of_your_video> -e <Name_of_model_exercise_>
```
*Note*: <Name_of_model_exercise> is the name of the video file choose without the video extension

## How to add exercises performed by a personal trainer?
1. Record a video and save it in the Data folder. Give a name to the file that you remember the exercise done<br>
2. In the directory move/models create a folder with the name of the exercise given to the video<br>
3. In the newly create folder insert a .txt file containing the parts of the body essential for the exercise. In the example file total_point.txt all parts of the body that you can insert in your .txt file are indicated<br>

To run the code for creating the model, insert to command line, inside the directory containing the files:
```sh
python Model_Acquisition.py -v <Path_of_file_video> -e <Name_of_model_exercise>
```
*Note*: <Name_of_model_exercise> is the name of your video file, insert into Data folder, without the video extension. This name is also the same as the folder created in move/models

## What's in the move folder?
All'interno della cartella move/models è possibile trovare delle sotto cartelle contenenti i modelli per ciascuna tipologia di esercizio. All'interno di ognuna di esse vi sono delle ulteriori cartelle divise in complete e cycle ed un file interest_point.txt contenente i giunti fondamentali per quel determinato esercizio. In complete è possibile trovare il file .csv contenente tutti i frame del video modello con indicate ciascuno di essi le rispettive coordinate x,y dei giunti, lo score di accuratezza ed la label con i nomi dei giunti. In cycle, invece, sono presenti i files che caratterizzano la posa modello: model.csv contiene i frame ed i parametri dei giunti relativi alla posa iniziale, individauta come modello; medium.csv contiene i frame e le distanze coseno medie calcolate su tutte le pose, per ciascun giunto e variance.csv, con le relative varianze.

## Run the code with/without sampling 
Di default il flag per il campionamento dei cicli è settato a false, ma è possibile variarlo all'interno dei file: 'Model_Acquisition.py' e 'Match.py'.

## Requirements
The use of a GPU is recommended for processing performed by Openpose. We used a * GTX 1060 *

## Contributors

The entire project was made by [Giovanna Scaramuzzino](https://github.com/ScaramuzzinoGiovanna) and [Johan Bosso](https://github.com/jabosso)

