train:
	python -u train.py | tee logfile

all:
	fetch train 

fetch:
	wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
	unzip trainingandtestdata.zip
	mv training.1600000.processed.noemoticon.csv training.csv
	mv testdata.manual.2009.06.14.csv testing.csv


clean:
	- rm loss_function.png error.png 
	- rm logfile

purge:
	- rm -rf ./checkpoints/*
	- rm training.csv
	- rm testing.csv
	- rm trainingandtestdata.zip
