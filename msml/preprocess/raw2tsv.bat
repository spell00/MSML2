@ECHO OFF
set spd=%1%
set experiment=%2%
set path=%3%
IF [%1%]==[] (set spd=200)
IF [%2%]==[] (set experiment=old_data)
IF [%3%]==[] (set path=..\resources)
dir /b "%path%\%experiment%\raw\%spd%spd\*.raw" > nul
FOR /R "%path%\%experiment%\raw\%spd%spd\" %%G IN (*.raw) DO (
    START .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "%path%\%experiment%\raw\%spd%spd\%%~nG.mzDB" -f 1-2 -a "dia")
GOTO TEST

:TEST
tasklist.exe | findstr "raw2mzDB.exe" > nul
cls
if errorlevel 1 ( GOTO NEXT ) else ( CALL timeout 10 /nobreak > nul && GOTO TEST )

:NEXT
if not exist "%path%\%experiment%\mzdb\%spd%spd\" MD %path%\%experiment%\mzdb\%spd%spd
FOR /R %path%\%experiment%\raw\%spd%spd %%G IN (*.mzdb) DO (
    MOVE "%path%\%experiment%\raw\%spd%spd\%%~nG.mzDB" "%path%\%experiment%\mzdb\%spd%spd\%%~nG.mzDB")

bash mzdb2tsv.sh