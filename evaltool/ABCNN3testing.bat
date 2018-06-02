@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\ABCNN3testing.score.txt
@set Output_metricResultFile=.\ABCNN3testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause