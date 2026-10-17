@echo off
setlocal

set VENV_PYTHON=c:\Users\fishministrator\Projects\dlgn_research\.venv\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    echo Using CUDA-enabled PyTorch environment from dlgn_research: %VENV_PYTHON%
    "%VENV_PYTHON%" run_suite.py %*
) else (
    echo Using system Python:
    python run_suite.py %*
)

endlocal
