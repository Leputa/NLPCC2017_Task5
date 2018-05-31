@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\ABCNN2testing.score.txt
@set Output_metricResultFile=.\ABCNN2testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause