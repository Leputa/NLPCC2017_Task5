@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\ABCNN1testing.score.txt
@set Output_metricResultFile=.\ABCNN1testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause