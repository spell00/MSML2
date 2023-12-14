@ECHO OFF
setlocal EnableDelayedExpansion

set "startTime=%time: =0%"
set experiment=%1%
set ms_level=%2%
set spd=%3%
set mz_bin=%4%
set rt_bin=%5%
set mz_bin_post=%6%
set rt_bin_post=%7%
set split_train_data=%8%
set feature_selection=%9%
SHIFT
SHIFT
SHIFT
set run_name=%7%
set n_trials=%8%
set test=%9%

IF [%experiment%]==[] (set experiment=old_data)
IF [%ms_level%]==[] (set ms_level=1)
IF [%spd%]==[] (set spd=300)
IF [%mz_bin%]==[] (set mz_bin=0.2)
IF [%rt_bin%]==[] (set rt_bin=20)
IF [%mz_bin_post%]==[] (set mz_bin_post=0.2)
IF [%rt_bin_post%]==[] (set rt_bin_post=20)
IF [%split_train_data%]==[] (set split_train_data=1)
IF [%feature_selection%]==[] (set feature_selection=mutual_info_classif)
IF [%run_name%]==[] (set run_name=eco,sag,efa,kpn,blk,pool)
IF [%n_trials%]==[] (set n_trials=1)
IF [%test%]==[] (set test=0)

if [%split_train_data%==1] (GOTO SPLIT)
if [%split_train_data%==0] (GOTO NOSPLIT)

:NOSPLIT
FOR /R "..\..\..\..\resources\%experiment%\raw\%spd%spd\train\" %%G IN (*.raw) DO (
    START /wait .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\%experiment%\raw\%spd%spd\train\%%~nG.mzDB" -f 1-2 -a "dia")
FOR /R "..\..\..\..\resources\%experiment%\raw\%spd%spd\valid\" %%G IN (*.raw) DO (
    START /wait .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\%experiment%\raw\%spd%spd\valid\%%~nG.mzDB" -f 1-2 -a "dia")
FOR /R "..\..\..\..\resources\%experiment%\raw\%spd%spd\test\" %%G IN (*.raw) DO (
    START /wait .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\%experiment%\raw\%spd%spd\test\%%~nG.mzDB" -f 1-2 -a "dia")

GOTO TEST_SPLIT

:SPLIT
PRINT 'SPLIT'
FOR /R "..\..\..\..\resources\%experiment%\raw\%spd%spd\" %%G IN (*.raw) DO (
    START .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\%experiment%\raw\%spd%spd\%%~nG.mzDB" -f 1-2 -a "dia")
GOTO TEST_NOSPLIT

:TEST_SPLIT
tasklist.exe | findstr "raw2mzDB.exe" > nul
cls
if errorlevel 1 ( GOTO NEXT_SPLIT ) else ( CALL timeout 10 /nobreak > nul && GOTO TEST_SPLIT )

:TEST_NOSPLIT
tasklist.exe | findstr "raw2mzDB.exe" > nul
cls
if errorlevel 1 ( GOTO NEXT_NOSPLIT ) else ( CALL timeout 10 /nobreak > nul && GOTO TEST_NOSPLIT )

:NEXT_SPLIT
if not exist "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\" MD ..\..\..\..\resources\%experiment%\mzdb\%spd%spd
FOR /R ..\..\..\..\resources\%experiment%\raw\%spd%spd %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\%experiment%\raw\%spd%spd\%%~nG.mzDB" "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\%%~nG.mzDB")

:NEXT_NOSPLIT
if not exist "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\train" MD ..\..\..\..\resources\%experiment%\mzdb\%spd%spd\train
if not exist "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\valid" MD ..\..\..\..\resources\%experiment%\mzdb\%spd%spd\valid
if not exist "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\test" MD ..\..\..\..\resources\%experiment%\mzdb\%spd%spd\test
FOR /R ..\..\..\..\resources\%experiment%\raw\%spd%spd\train %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\%experiment%\raw\%spd%spd\train\%%~nG.mzDB" "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\train\%%~nG.mzDB")
FOR /R ..\..\..\..\resources\%experiment%\raw\%spd%spd\valid %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\%experiment%\raw\%spd%spd\valid\%%~nG.mzDB" "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\valid\%%~nG.mzDB")
FOR /R ..\..\..\..\resources\%experiment%\raw\%spd%spd\test %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\%experiment%\raw\%spd%spd\test\%%~nG.mzDB" "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\test\%%~nG.mzDB")

rem This is the time to transform raw files into mzdb files
set "endTime=%time: =0%"
rem Get elapsed time:
set "end=!endTime:%time:~8,1%=%%100)*100+1!"  &  set "start=!startTime:%time:~8,1%=%%100)*100+1!"
set /A "elap=((((10!end:%time:~2,1%=%%100)*60+1!%%100)-((((10!start:%time:~2,1%=%%100)*60+1!%%100), elap-=(elap>>31)*24*60*60*100"

rem Convert elapsed time to HH:MM:SS:CC format:
set /A "cc=elap%%100+100,elap/=100,ss=elap%%60+100,elap/=60,mm=elap%%60+100,hh=elap/60+100"
echo Elapsed (raw2mzdb):  %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1%

bash msml/preprocess/mzdb2tsv.sh %mz_bin% %rt_bin% %spd% %ms_level% %experiment% %split_train_data%

IF %split_train_data%==1 (
    IF %ms_level%==1 (GOTO MAKE_TENSORS_MS1_SPLIT)
    IF %ms_level%==2 (GOTO MAKE_TENSORS_MS2_SPLIT)
)
IF %split_train_data%==0 (
    IF %ms_level%==1 (GOTO MAKE_TENSORS_MS1)
    IF %ms_level%==2 (GOTO MAKE_TENSORS_MS2)
)

:MAKE_TENSORS_MS1_SPLIT
python msml/preprocess/make_tensors_ms1_split.py --test_run=%test_run% --run_name=%run_name% --feature_selection=%feature_selection% --mz_bin_post=%mz_bin_post% --rt_bin_post=%rt_bin_post% --mz_bin=%mz_bin% --rt_bin=%rt_bin%
GOTO TRAIN_AE

:MAKE_TENSORS_MS2_SPLIT
python msml/preprocess/make_tensors_ms2_split.py --test_run=%test_run% --run_name=%run_name% --feature_selection=%feature_selection% --mz_bin_post=%mz_bin_post% --rt_bin_post=%rt_bin_post% --mz_bin=%mz_bin% --rt_bin=%rt_bin%
GOTO TRAIN_AE

:MAKE_TENSORS_MS1
python msml/preprocess/make_tensors_ms1.py --test_run=%test_run% --run_name=%run_name% --feature_selection=%feature_selection% --mz_bin_post=%mz_bin_post% --rt_bin_post=%rt_bin_post% --mz_bin=%mz_bin% --rt_bin=%rt_bin%
GOTO TRAIN_AE

:MAKE_TENSORS_MS2
python msml/preprocess/make_tensors_ms2.py --test_run=%test_run% --run_name=%run_name% --feature_selection=%feature_selection% --mz_bin_post=%mz_bin_post% --rt_bin_post=%rt_bin_post% --mz_bin=%mz_bin% --rt_bin=%rt_bin%
GOTO TRAIN_AE

rem This is the time to transform raw files into df files
set "endTime=%time: =0%"
rem Get elapsed time:
set "end=!endTime:%time:~8,1%=%%100)*100+1!"  &  set "start=!startTime:%time:~8,1%=%%100)*100+1!"
set /A "elap=((((10!end:%time:~2,1%=%%100)*60+1!%%100)-((((10!start:%time:~2,1%=%%100)*60+1!%%100), elap-=(elap>>31)*24*60*60*100"

echo Elapsed (raw2df):  %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1%
