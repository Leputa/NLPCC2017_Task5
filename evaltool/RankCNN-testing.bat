@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\RankCNN-testing.score.txt
@set Output_metricResultFile=.\RankCNN-testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause