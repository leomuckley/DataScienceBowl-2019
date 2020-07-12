# DataScienceBowl-2019
Python code for the Data Science Bowl 2019 competition (top 7%) hosted by Kaggle

In this challenge, anonymous gameplay data was used, including knowledge of videos watched and games played, 
from the PBS KIDS Measure Up! app, a game-based learning tool developed as a part of the CPB-PBS Ready To Learn Initiative
with funding from the U.S. Department of Education. 
Competitors were challenged to predict scores on in-game assessments and create an algorithm that will lead to better-designed 
games and improved learning outcomes.

The intent of the competition is was to use the gameplay data to forecast how many attempts a child will take to pass a given
assessment (an incorrect answer is counted as an attempt). The outcomes in this competition are grouped into 4 groups 
(labeled  `<em>accuracy_group</em>`  in the data):

3: the assessment was solved on the first attempt<br/>
2: the assessment was solved on the second attempt<br/>
1: the assessment was solved after 3 or more attempts<br/>
0: the assessment was never solved<br/>

My solution involved using a regression approach to this task where the evaluation was based on kappa metric. 
Conisdering the ordinal form of the target feature, a regression approach with optimised thresholds were implemented.
