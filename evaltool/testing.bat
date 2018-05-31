@set Input_qaPariFile=.\testing.QApair.txt
@set Input_resultFile=.\testing.score.txt
@set Output_metricResultFile=.\testing.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause