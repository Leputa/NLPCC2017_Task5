@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\ABCNN3-testing.score.txt
@set Output_metricResultFile=.\ABCNN3-testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause