Installazioni necessarie: 
-opencv
-pyopenpose
-scipy
-numpy
-math
-argparse
-time

All'interno della cartella move/models è possibile trovare delle sotto cartelle contenenti i modelli per ciascuna tipologia di esercizio. All'interno di ognuna di esse vi sono delle ulteriori cartelle divise in complete e cycle ed un file interest_point.txt contenente i giunti fondamentali per quel determinato esercizio. In complete è possibile trovare il file .csv contenente tutti i frame del video modello con indicate ciascuno di essi le rispettive coordinate x,y dei giunti, lo score di accuratezza ed la label con i nomi dei giunti. In cycle, invece, sono presenti i files che caratterizzano la posa modello: model.csv contiene i frame ed i parametri dei giunti relativi alla posa iniziale, individauta come modello; medium.csv contiene i frame e le distanze coseno medie calcolate su tutte le pose, per ciascun giunto e variance.csv, con le relative varianze.

Per la creazione di un nuovo modello è necessaria la creazione di un file .txt contenente le parti del corpo fondamentali per l'esercizio, come indicate nel file di esempio total_point.txt.

Tale file deve essere inserito in una cartella che deve avere il nome relativo all'esercizio, all'interno di move/models.
Per avviare la creazione del modello deve essere inserito da linea di comando, all'interno della directory in cui sono contenuti i file: 
python Model_Acquisition.py -v _Path_File_Video_ -e _Nome_Esercizio_ 
Il codice è stato scritto in modo tale da acquisire solo video ripresi tramite file .mp4 e non tramite webcam, in modo tale da avere maggior qualità video ed avere così un maggior numero di frame su cui poter eseguire l'elaborazione.
Inoltre, poichè opencv non legge i metadati del video, è opportuna una ripresa landscape.

Per confrontare un nuovo video, con uno di quelli modello, è necessario inserire da linea di comando:
python Match.py -v _Path_File_Video_ -e _Nome_Esercizio_ 
Il nome dell'esercizio inserito deve essere corrispondete ad uno di quelli presenti nella cartella move/models.

Per il confronto con i modelli da noi creati, è possibile trovare tali video, nella cartella Data (file zip da scompattare), con sottodirectory alcune tipologie di esercizio, al cui interno vi sono i file video, da visionare prima di eseguire l'esercizio.

Di default il flag per il campionamento dei cicli è settato a false, ma è possibile variarlo all'interno dei file: 'Model_Acquisition.py' e 'Match.py'.







