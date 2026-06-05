@echo off
setlocal
g++ -std=c++17 -O2 -Iinclude -I. src\main.cpp src\api_server.cpp -o db -lws2_32
if %ERRORLEVEL% EQU 0 (
    echo Build OK: db.exe
) else (
    echo Build failed.
    exit /b 1
)
