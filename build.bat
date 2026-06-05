@echo off
setlocal

where g++ >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo g++ was not found. Install MSYS2 / MinGW and add g++ to PATH.
    exit /b 1
)

echo Building MY-AI...
g++ -std=c++17 -O2 -Iinclude -I. src\main.cpp src\api_server.cpp -o db.exe -lws2_32
if %ERRORLEVEL% EQU 0 (
    echo Build OK: db.exe
) else (
    echo Build failed.
    exit /b 1
)
